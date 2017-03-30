/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI.h>
#include "sptensor.h"
#include "../cudawrap.h"
#include "mttkrp_cuda_kernels.h"



/**
 * CUDA parallelized Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 * In this version, atomic function to lock the global reduction and a large
 * scratch is used to maximize parallelism. (To be optimized)
 */
int sptCudaMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    size_t * const mats_order,    // Correspond to the mode order of X.
    size_t const mode,
    size_t const impl_num) 
{
    size_t const nmodes = X->nmodes;
    size_t const nnz = X->nnz;
    size_t const * const ndims = X->ndims;
    size_t const R = mats[mode]->ncols;
    size_t const stride = mats[mode]->stride;
    int result;

    double time_h2d, time_exe, time_d2h;
    double gbw_h2d, gflops_exe, gbw_d2h;
    sptTimer timer;
    sptNewTimer(&timer, 0);

    /* Check the mats. */
    for(size_t i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }


    /* Transfer tensor and matrices */
    /* dev_mats_order: 1st gpu. */
    size_t * dev_mats_order;
    /* dev_Xndims: 1st gpu. */
    size_t * dev_Xndims;
    /* dev_Xvals: 1st gpu. */
    sptScalar * dev_Xvals;
    /* Xinds_header: 1st cpu, 2nd cpu (ghost pointers) */
    size_t ** Xinds_header = new size_t *[nmodes];
    /* dev_Xinds: 1st gpu, 2nd gpu. */
    size_t ** dev_Xinds;
    /* mats_header: 1st cpu, 2nd cpu (ghost pointers) */
    sptScalar ** mats_header = new sptScalar *[nmodes+1];
    /* lengths: 1st cpu, store the lengths of mats */
    size_t * const lengths = new size_t[nmodes+1];
    /* dev_mats: 1st gpu, 2nd gpu. */
    sptScalar ** dev_mats;
    /* dev_scratch: 1st gpu. */
    sptScalar * dev_scratch;
    /* the pointer to dev_mats[nmodes] */
    sptScalar *dev_part_prod;  
    size_t dev_mem_size = 0;
    size_t dev_flops = 2 * nnz * R + (nmodes - 1) * R;


    sptStartTimer(timer);

    /* dev_mats_order */
    result = sptCudaDuplicateMemory(&dev_mats_order, mats_order, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nmodes * sizeof (size_t);

    /* dev_Xndims */
    result = sptCudaDuplicateMemory(&dev_Xndims, ndims, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nmodes * sizeof (size_t);

    /* dev_Xvals */
    result = sptCudaDuplicateMemory(&dev_Xvals, X->values.data, nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nnz * sizeof (sptScalar);

    /* Xinds_header */
    for(size_t m = 0; m < nmodes; ++m) {
        Xinds_header[m] = X->inds[m].data;
    }
    /* dev_Xinds */
    result = sptCudaDuplicateMemoryIndirect(&dev_Xinds, Xinds_header, nmodes, nnz, cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nmodes * nnz * sizeof(size_t);

    /* mats_header and lengths */
    size_t sum_mat_length = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        mats_header[m] = mats[m]->values;
        lengths[m] = mats[m]->nrows * stride;
        sum_mat_length += mats[m]->nrows * stride;
    }
    mats_header[nmodes] = mats[nmodes]->values;
    lengths[nmodes] = mats[mode]->nrows * stride;
    sum_mat_length += mats[mode]->nrows * stride;
    /* dev_mats */
    result = sptCudaDuplicateMemoryIndirect(&dev_mats, mats_header, nmodes+1, lengths, cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += sum_mat_length * sizeof(sptScalar);

    if(nmodes > 4) {
        /* dev_scratch */
        result = cudaMalloc((void **) &dev_scratch, nnz * stride * sizeof (sptScalar));
        spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
        result = cudaMemset(dev_scratch, 0, nnz * stride * sizeof (sptScalar));
        spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
        dev_mem_size +=  nnz * stride * sizeof (sptScalar);
    }

    sptStopTimer(timer);
    time_h2d = sptElapsedTime(timer);
    gbw_h2d = dev_mem_size / time_h2d /1e9;
    sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP H2D");
    printf("[Bandwidth H2D]: %lf GBytes/sec\n", gbw_h2d);


    // size_t max_nthreads_per_block = 512;    // old run
    size_t max_nthreads_per_block = 256;
    size_t max_nblocks = 32768;
    size_t max_nthreadsy = 16;

    size_t nthreadsx = 0;
    size_t nthreadsy = 0;
    size_t all_nblocks = 0;
    switch(impl_num) {
    case 1: // Naive, 1D
        nthreadsx = 256;
        all_nblocks = (nnz + nthreadsx -1) / nthreadsx;
        break;
    case 2: // 2D
        nthreadsy = R;
        nthreadsx = max_nthreads_per_block / nthreadsy;
        all_nblocks = (nnz + nthreadsx -1) / nthreadsx;
        break;
    case 3: // 2D, rank split
        if(R <= max_nthreadsy)
            nthreadsy = R;
        else
            nthreadsy = max_nthreadsy;
        nthreadsx = max_nthreads_per_block / nthreadsy;
        all_nblocks = (nnz + nthreadsx -1) / nthreadsx;
        break;
    case 4: // 2D, exchange x and y
        nthreadsx = R;
        nthreadsy = max_nthreads_per_block / nthreadsx;
        all_nblocks = (nnz + nthreadsy -1) / nthreadsy;
        break;
    case 5: // 2D, exchange x and y, rank split. Best performance
        if(R <= max_nthreadsy)
            nthreadsx = R;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;
        all_nblocks = (nnz + nthreadsy -1) / nthreadsy;
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %zu, nthreadsx: %zu, nthreadsy: %zu\n", all_nblocks, nthreadsx, nthreadsy);


  sptStartTimer(timer);

  for(size_t block_offset = 0; block_offset < all_nblocks; block_offset += max_nblocks) {
    printf("block_offset: %zu\n", block_offset);
    size_t nblocks = (all_nblocks >= block_offset) ? all_nblocks - block_offset: 0;
    if(nblocks > max_nblocks) {
        nblocks = max_nblocks;
    }


    switch(nmodes) {
    case 3:
        switch(impl_num) {
        case 1: // Naive
            printf("Execute spt_MTTKRPKernelNnz3D (%zu, %zu)\n", nblocks, nthreadsx);
            spt_MTTKRPKernelNnz3D<<<nblocks, nthreadsx>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats,
                block_offset);
            break;
        case 2:
            printf("Execute spt_MTTKRPKernelNnzRank3D (%zu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            spt_MTTKRPKernelNnzRank3D<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats,
                block_offset);
            break;
        case 3:
            printf("Execute spt_MTTKRPKernelNnzRankSplit3D (%zu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            spt_MTTKRPKernelNnzRankSplit3D<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats,
                block_offset);
            break;
        case 4:
            printf("Execute spt_MTTKRPKernelRankNnz3D (%zu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            spt_MTTKRPKernelRankNnz3D<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats,
                block_offset);
            break;
        case 5:
            printf("Execute spt_MTTKRPKernelRankSplitNnz3D (%zu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            spt_MTTKRPKernelRankSplitNnz3D<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats,
                block_offset);
            break;
        }   // End switch impl_num
        break;

    case 4: 
        switch(impl_num) {
        default:
            printf("Not support: Execute spt_MTTKRPKernelScratch (%zu, %zu)\n", nblocks, nthreadsx);
            // spt_MTTKRPKernelScratch<<<nblocks, nthreadsx>>>(
            //     mode,
            //     nmodes,
            //     nnz,
            //     R,
            //     stride,
            //     dev_Xndims,
            //     dev_Xinds,
            //     dev_Xvals,
            //     dev_mats_order,
            //     dev_mats,
            //     dev_scratch,
            //     block_offset);
        }   // End switch impl_num
        break;

    default:
        printf("Execute spt_MTTKRPKernelScratch (%zu, %zu)\n", nblocks, nthreadsx);
        spt_MTTKRPKernelScratch<<<nblocks, nthreadsx>>>(
            mode,
            nmodes,
            nnz,
            R,
            stride,
            dev_Xndims,
            dev_Xinds,
            dev_Xvals,
            dev_mats_order,
            dev_mats,
            dev_scratch,
            block_offset);
    }   // End switch nmodes
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  } // End loop block_offset


    sptStopTimer(timer);
    time_exe = sptElapsedTime(timer);
    gflops_exe = dev_flops / time_exe / 1e9;
    sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP");
    printf("[GFLOPS]: %lf GFlops \n", gflops_exe);

    sptStartTimer(timer);

    dev_mem_size = 0;
    /* Copy back the pointer to dev_mats[nmodes] to the result */
    result = cudaMemcpy(&dev_part_prod, dev_mats + nmodes, sizeof dev_part_prod, cudaMemcpyDeviceToHost);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += sizeof dev_part_prod;

    result = cudaMemcpy(mats[nmodes]->values, dev_part_prod, mats[mode]->nrows * stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += mats[mode]->nrows * stride * sizeof (sptScalar);

    sptStopTimer(timer);
    time_d2h = sptElapsedTime(timer);
    gbw_d2h = dev_mem_size / time_d2h /1e9;
    sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP D2H");
    printf("[Bandwidth D2H]: %lf GBytes/sec\n", gbw_d2h);
    sptFreeTimer(timer);

    result = cudaFree(dev_mats_order);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_Xndims);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_Xvals);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_Xinds);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_mats);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    if(nmodes > 4) {
        result = cudaFree(dev_scratch);
        spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    }
    delete[] Xinds_header;
    delete[] mats_header;
    delete[] lengths;

  return 0;
}



int sptCudaMTTKRPDevice(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t rank,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch)
{
  int result;

  result = cudaMemset(dev_scratch, 0, nnz * rank * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  size_t nthreads = 128;
  const size_t max_nblocks = 32768;
  size_t all_nblocks = (nnz + nthreads -1) / nthreads;

  // sptTimer timer;
  // sptNewTimer(&timer, 0);
  // sptStartTimer(timer);


  for(size_t block_offset = 0; block_offset < all_nblocks; block_offset += max_nblocks) {
    size_t nblocks = all_nblocks - block_offset;
    if(nblocks > max_nblocks) {
        nblocks = max_nblocks;
    }
    spt_MTTKRPKernelScratch<<<nblocks, nthreads>>>(
        mode,
        nmodes,
        nnz,
        rank,
        stride,
        Xndims,
        Xinds,
        Xvals,
        dev_mats_order,
        dev_mats,
        dev_scratch,
        block_offset
        );
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }

  // sptStopTimer(timer);
  // sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP");
  // sptFreeTimer(timer);


  return 0;
}

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
#include <cuda_runtime.h>
#include "../cudawrap.h"



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
int sptCudaCoarseSMMTTKRP(
    double *queue_time,
    sptSparseTensor * const tsr,
    spt_SplitResult const *splits,
    size_t const queue_size,
    size_t const nblocks,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode) 
{
    #if 0
    if(queue_size == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "queue_size == 0");
    }
    if(nblocks == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "nblocks == 0");
    }

    /* nmodes, R, and stride are the same for all sub-tensors */
    size_t nmodes = splits->tensor.nmodes;
    size_t R = mats[mode]->ncols;
    size_t stride = mats[mode]->stride;
    int result;

    /* Check the mats. */
    for(size_t i = 0; i < nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
    }

    /* Initialize result matrix */
    sptConstantMatrix(mats[nmodes], 0); 

    /* dev_mats_order[i, m] <= mats_order[i, m] */
    size_t *dev_mats_order;
    result = sptCudaDuplicateMemory(&dev_mats_order, mats_order, nmodes * sizeof *dev_mats_order, cudaMemcpyHostToDevice);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

    /* Very careful about these sizes */
    size_t ** dev_nnz = new size_t *[nstreams];
    size_t ** dev_Xndims = new size_t *[nstreams];
    size_t ** Xinds_header = new size_t *[nmodes];
    size_t *** dev_Xinds = new size_t **[nstreams];
    sptScalar ** dev_Xvals = new sptScalar *[nstreams];
    sptScalar ** mats_header = new sptScalar *[nmodes+1];
    size_t * const lengths = new size_t[nmodes+1];
    sptScalar *** dev_mats = new sptScalar **[nstreams];
    /* Initialize part_prod matrix, to store partial results of updated mats[mode] */
    sptMatrix * part_prod = (sptMatrix*)malloc(nstreams * sizeof(sptMatrix)); 
    
    for(size_t i = 0; i < nstreams; ++i) {
        result = cudaMalloc(&dev_Xndims[i], nmodes * sizeof *dev_Xndims);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    }

    const spt_SplitResult * splits_loop = splits;
    double elapsed_time = 0;
    sptTimer timer;
    sptNewTimer(&timer, 0);

    size_t nstreams = (queue_size-1)/nblocks + 1;
    size_t stream_count = 0;
    size_t sum_nnz = 0, nnz_stream_begin = 0;
    while(splits_loop != NULL && stream_count < nstreams) {
        size_t * min_inds_low = (size_t *)malloc(nmodes * sizeof(size_t));
        size_t * max_inds_high = (size_t *)malloc(nmodes * sizeof(size_t));
        size_t * blocks_nnz = (size_t *)malloc(nblocks * sizeof(size_t));
        for(size_t m=0; m<nmodes; ++m) {
            min_inds_low[m] = tsr->ndims[m];
            max_inds_high[m] = 0;
        }

        result = cudaMemcpy(dev_Xndims[stream_count], splits_loop->tensor.ndims, nmodes * sizeof *dev_Xndims, cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        /* Get the index range and nnz number for each kernel */
        const spt_SplitResult * splits_tmp = splits_loop;
        size_t b = 0;
        while(splits_tmp != NULL && b < nblocks) {
            blocks_nnz[b] = splits_tmp->tensor.nnz;
            sum_nnz += blocks_nnz[b];
            for(size_t m = 0; m < nmodes; ++m) {
                if(min_inds_low[m] > splits_tmp->inds_low[m]) {
                    min_inds_low[m] = splits_tmp->inds_low[m];
                }
                if(max_inds_high[m] < splits_tmp->inds_high[m]) {
                    max_inds_high[m] = splits_tmp->inds_high[m];
                }
            }
            ++ b;
            splits_tmp = splits_tmp->next;
        }

        for(size_t m = 0; m < nmodes; ++m) {
            mats_header[m] = mats[m]->values + min_inds_low[m];
            lengths[m] = (max_inds_high[m] - min_inds_low[m]) * stride;
        }
        mats_header[nmodes] = mats[nmodes]->values;
        lengths[nmodes] = sptGetMatrixLength(mats[mode]);
        result = sptCudaDuplicateMemoryIndirect(&dev_mats[stream_count], mats_header, nmodes+1, lengths, cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        result = sptCudaDuplicateMemory(&dev_nnz[stream_count], blocks_nnz, nblocks * sizeof(size_t), cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        for(size_t m = 0; m < nmodes; ++m) {
            Xinds_header[m] = tsr->inds[m].data + nnz_stream_begin;
        }
        result = sptCudaDuplicateMemoryIndirect(&dev_Xinds[stream_count], Xinds_header, nmodes, sum_nnz, cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        result = sptCudaDuplicateMemory(&dev_Xvals[stream_count], splits_loop->tensor.values.data + nnz_stream_begin, sum_nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        sptStartTimer(timer);
        size_t nthreads = 128;
        spt_MTTKRPKernel<<<nblocks, nthreads>>>(
            mode,
            nmodes,
            dev_nnz[stream_count],
            R,
            stride,
            dev_Xndims[stream_count],
            dev_Xinds[stream_count],
            dev_Xvals[stream_count],
            dev_mats_order[stream_count],
            dev_mats[stream_count],
            dev_scratch[stream_count]
        );
        result = cudaDeviceSynchronize();
        spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

        sptStopTimer(timer);
        elapsed_time += sptElapsedTime(timer);


        result = sptNewMatrix(&part_prod[i], (max_inds_high[mode] - min_inds_low[mode]), mats[mode]->ncols);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        sptScalar *dev_part_prod;
        cudaMemcpy(&dev_part_prod, dev_mats[stream_count] + nmodes, sizeof dev_part_prod, cudaMemcpyDeviceToHost);

        result = cudaMemcpy(part_prod.values, dev_part_prod, sptGetMatrixLength(&part_prod) * sizeof (sptScalar), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

        for(size_t i = 0; i < sptGetMatrixLength(&part_prod); ++i) {
            mats[nmodes]->values[i] += part_prod.values[i];
        }

        result = cudaMemset(dev_part_prod, 0, sptGetMatrixLength(&part_prod) * sizeof (sptScalar));
        spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");


        /* Next stream */
        nnz_stream_begin += sum_nnz;
        ++ stream_count;
        splits_loop = splits_loop->next;
    }   // End while(splits_loop != NULL && stream_count < nstreams) 

#endif


    


    // size_t const nmodes = X->nmodes;
    // size_t const nnz = X->nnz;
    // size_t const * const ndims = X->ndims;
    // size_t const R = mats[mode]->ncols;
    // size_t const stride = mats[mode]->stride;
    // size_t const nmats = nmodes - 1;
    // int result;

    // /* Check the mats. */
    // for(size_t i=0; i<nmodes; ++i) {
    //     if(mats[i]->ncols != mats[nmodes]->ncols) {
    //         spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    //     }
    //     if(mats[i]->nrows != ndims[i]) {
    //         spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    //     }
    // }


    // /* Transfer tensor and matrices */
    // size_t * dev_Xndims = NULL;
    // result = sptCudaDuplicateMemory(&dev_Xndims, ndims, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

    // sptScalar * dev_Xvals = NULL;
    // result = sptCudaDuplicateMemory(&dev_Xvals, X->values.data, nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


    // size_t ** Xinds_header = (size_t**)malloc(nmodes*sizeof(size_t*));
    // for(size_t m = 0; m < nmodes; ++m) {
    //     Xinds_header[m] = X->inds[m].data;
    // }
    // size_t ** dev_Xinds = NULL;   // array of pointer to device memory
    // result = sptCudaDuplicateMemoryIndirect(&dev_Xinds, Xinds_header, nmodes, nnz, cudaMemcpyHostToDevice);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

    // size_t * dev_mats_order = NULL;
    // result = sptCudaDuplicateMemory(&dev_mats_order, mats_order->data, nmats * sizeof (size_t), cudaMemcpyHostToDevice);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

    // sptScalar ** tmp_mats = NULL;
    // tmp_mats = (sptScalar **)malloc((nmodes+1) * sizeof(sptScalar*));
    // for(size_t i=0; i<nmodes+1; ++i) {
    // result = cudaMalloc((void **) &(tmp_mats[i]), mats[i]->nrows * mats[i]->stride * sizeof(sptScalar));
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // result = cudaMemcpy(tmp_mats[i], mats[i]->values, mats[i]->nrows * mats[i]->stride * sizeof(sptScalar), cudaMemcpyHostToDevice);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // }
    // result = cudaMemset(tmp_mats[nmodes], 0, mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar));
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // sptScalar ** dev_mats = NULL;   // array of pointer to device memory
    // result = cudaMalloc((void ***) &dev_mats, (nmodes+1) * sizeof(sptScalar*));
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // result = cudaMemcpy(dev_mats, tmp_mats, (nmodes+1) * sizeof (sptScalar*), cudaMemcpyHostToDevice);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");



    // const size_t nthreads = 128;
    // const size_t max_nblocks = 32768;
    // size_t all_nblocks = (nnz + nthreads -1) / nthreads;
    // printf("all_nblocks: %lu, nthreads: %lu\n", all_nblocks, nthreads);

    // sptTimer timer;
    // sptNewTimer(&timer, 0);
    // sptStartTimer(timer);

    // for(size_t block_offset = 0; block_offset < all_nblocks; block_offset += max_nblocks) {
    // size_t nblocks = all_nblocks - block_offset;
    // if(nblocks > max_nblocks) {
    //     nblocks = max_nblocks;
    // }
    // spt_CoarseMTTKRPKernel<<<nblocks, nthreads>>>(
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
    //     block_offset
    //     );
    // result = cudaThreadSynchronize();
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // }


    // sptStopTimer(timer);
    // sptPrintElapsedTime(timer, "CUDA SpTns Coarse MTTKRP");
    // sptFreeTimer(timer);


    // result = cudaMemcpy(mats[nmodes]->values, tmp_mats[nmodes], mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP copy back");


    // for(size_t i=0; i<nmodes+1; ++i) {
    //     result = cudaFree(tmp_mats[i]);
    //     spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // }
    // result = cudaFree(dev_mats);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

    // result = cudaFree(dev_Xndims);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // result = cudaFree(dev_Xvals);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // result = cudaFree(dev_mats_order);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // result = cudaFree(dev_Xinds);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // result = cudaFree(dev_mats);
    // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    // free(Xinds_header);
    // free(tmp_mats);

    return 0;
}

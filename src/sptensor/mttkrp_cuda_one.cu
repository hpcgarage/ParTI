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
 * Do NOT initialize result matrix, mats[nmodes], inside.
 */
int sptCudaOneMTTKRP(
    double *queue_time,
    double *queue_time_h2d,
    double *queue_time_d2h,
    double *queue_time_reduce,
    int const split_grain,
    sptSparseTensor * const tsr,
    spt_SplitResult const *splits,
    size_t const queue_size,
    size_t const nblocks,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    size_t const nnz_split_begin,
    size_t const max_nstreams,
    size_t const max_nthreadsy,
    size_t const smem_size,
    size_t const impl_num,
    size_t const cuda_dev_id) 
{
    if(queue_size == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "queue_size == 0");
    }
    if(nblocks == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "nblocks == 0");
    }
    cudaSetDevice(cuda_dev_id);

    /* nmodes, R, and stride are the same for all sub-tensors */
    size_t const nmodes = splits->tensor.nmodes;
    size_t const R = mats[mode]->ncols;
    size_t const stride = mats[mode]->stride;
    int result;

    /* Check the mats. */
    for(size_t i = 0; i < nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
    }

    /* Initialize part_prod matrix, to store partial results of updated mats[nmodes] */
    sptMatrix part_prod;
    result = sptNewMatrix(&part_prod, mats[mode]->nrows, mats[mode]->ncols);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);


    /*** Very careful about these sizes!!! ***/
    /* dev_mats_order: 1st gpu. Shared by max_nstreams */
    size_t *dev_mats_order;
    /* min_inds_low: 1st cpu, 2nd cpu. Minimum low indices for each stream */
    size_t ** min_inds_low = new size_t *[max_nstreams];
    /* dev_min_inds_low: 1st cpu, 2nd gpu. Minimum low indices for each stream */
    size_t ** dev_min_inds_low = new size_t *[max_nstreams];
    /* max_inds_high: 1st cpu, 2nd cpu. Maximum high indices for each stream */
    size_t ** max_inds_high = new size_t *[max_nstreams];
    /* inds_low_header: 1st cpu, 2nd cpu (ghost pointers) */
    size_t **inds_low_header = new size_t *[nblocks];
    /* dev_inds_low: 1st cpu, 2nd gpu, 3rd gpu. nblocks subtsrs' inds_low per stream */
    size_t ***dev_inds_low = new size_t **[max_nstreams];
    /* Xndims_header: 1st cpu, 2nd cpu (ghost pointers) */
    size_t **Xndims_header = new size_t *[nblocks];
    /* dev_Xndims: 1st cpu, 2nd gpu, 3rd gpu. nblocks subtsrs' ndims per stream */
    size_t ***dev_Xndims = new size_t **[max_nstreams];
    /* nnz_blk: 1st cpu, save the copy of nnz of each thread block */
    size_t *nnz_blk = new size_t[nblocks];
    /* max_nnz_stream: 1st cpu, save the maximum nnz of each stream */
    size_t *max_nnz_stream = new size_t[max_nstreams];
    /* nblocks_count: 1st cpu, save the actual number of blocks of each stream */
    size_t *nblocks_count = new size_t[max_nstreams];
    /* dev_nnz: 1st cpu, 2nd gpu. nblocks subtsrs' nnzs per stream */
    size_t **dev_nnz = new size_t *[max_nstreams];
    /* nnz_blk_begin: 1st cpu, save the copy of nnz of each thread block */
    size_t *nnz_blk_begin = new size_t[nblocks];
    /* dev_nnz_blk_begin: 1st cpu, 2nd gpu. nblocks subtsrs' nnz begin locations per stream */
    size_t **dev_nnz_blk_begin = new size_t *[max_nstreams];
    /* mats_header: 1st cpu, 2nd cpu (ghost pointers) */
    sptScalar **mats_header = new sptScalar *[nmodes+1];
    /* lengths: 1st cpu, store the lengths of mats */
    size_t * const lengths = new size_t[nmodes+1];
    /* dev_mats: 1st cpu, 2nd gpu, 3rd gpu. One copy of subtsr's corresponding mats per gpu, shared by nblocks */
    sptScalar ***dev_mats = new sptScalar **[max_nstreams];
    /* dev_part_prod: 1st gpu, 2nd gpu, the pointer to dev_mats[stream_idx][nmodes] */
    sptScalar **dev_part_prod = new sptScalar*[max_nstreams];
    /* Xinds_header: 1st cpu, 2nd cpu (ghost pointers) */
    size_t **Xinds_header = new size_t *[nmodes];
    /* dev_Xinds: 1st cpu, 2nd gpu, 3rd gpu. One copy of subtsr's inds per gpu, using the minimum and maximum indices of nblocks */
    size_t ***dev_Xinds = new size_t **[max_nstreams];
    /* dev_Xvals: 1st cpu, 2nd gpu. One copy of subtsr's vals per gpu */
    sptScalar **dev_Xvals = new sptScalar *[max_nstreams];


    /* dev_mats_order */
    result = sptCudaDuplicateMemory(&dev_mats_order, mats_order, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

    for(size_t stream_idx = 0; stream_idx < max_nstreams; ++stream_idx) {
        min_inds_low[stream_idx] = new size_t[nmodes];
        max_inds_high[stream_idx] = new size_t[nmodes];
    }    

    size_t total_nstreams = (queue_size-1)/nblocks + 1;
    size_t rest_nblocks = queue_size - (total_nstreams - 1) * nblocks;
    size_t batched_nstreams = (total_nstreams-1)/max_nstreams + 1;
    printf("total_nstreams: %zu, batched_nstreams: %zu, rest_nblocks: %zu\n", total_nstreams, batched_nstreams, rest_nblocks);
    double elapsed_time = 0, time_h2d = 0, time_d2h = 0, time_reduce = 0;
    sptTimer timer;
    sptNewTimer(&timer, 0);
    size_t nnz_begin = nnz_split_begin;

    /* Loop batches of streams */
    for(size_t sbatch_idx = 0; sbatch_idx < batched_nstreams; ++sbatch_idx) {
        /* Index inside sbatch_idx */
        size_t stream_count = sbatch_idx == batched_nstreams-1 ? total_nstreams - sbatch_idx * max_nstreams : max_nstreams;
        printf("sbatch_idx: %zu, stream_count: %zu\n", sbatch_idx, stream_count); fflush(stdout);

        /* Loop cocurrent streams */
        for(size_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
            size_t sum_nnz = 0;
            for(size_t m=0; m<nmodes; ++m) {
                min_inds_low[stream_idx][m] = tsr->ndims[m];
                max_inds_high[stream_idx][m] = 0;
            }
            max_nnz_stream[stream_idx] = 0;
            nblocks_count[stream_idx] = 0;
            
            /* Loop blocks inside one stream, nblocks should allocate once to a GPU. */
            nnz_blk_begin[0] = 0;
            nblocks_count[stream_idx] = (stream_idx == stream_count - 1) ? rest_nblocks: nblocks;
            for(size_t block_idx = 0; block_idx < nblocks_count[stream_idx]; ++block_idx) 
            {
                size_t queue_idx = (sbatch_idx * max_nstreams + stream_idx) * nblocks + block_idx;
                // printf("sbatch_idx: %zu, stream_idx: %zu, block_idx: %zu, queue_idx: %zu\n", sbatch_idx, stream_idx, block_idx, queue_idx);
                sptAssert(queue_idx < queue_size);
                sptAssert(&(splits[queue_idx].tensor) != NULL);

                sptSparseTensor const * subtsr_ptr = &(splits[queue_idx].tensor);
                size_t * inds_low_ptr = splits[queue_idx].inds_low;
                size_t * inds_high_ptr = splits[queue_idx].inds_high;
                // printf("block_idx: %zu, inds_low_ptr, inds_high_ptr:\n", block_idx);
                // spt_DumpArray(inds_low_ptr, nmodes, 0, stdout);
                // spt_DumpArray(inds_high_ptr, nmodes, 0, stdout);

                inds_low_header[block_idx] = inds_low_ptr;
                Xndims_header[block_idx] = subtsr_ptr->ndims;
                nnz_blk[block_idx] = subtsr_ptr->nnz;
                if(block_idx > 0)
                    nnz_blk_begin[block_idx] = nnz_blk_begin[block_idx-1] + nnz_blk[block_idx-1]; 

                /* Determine the needed range of factor matrices */
                for(size_t m=0; m<nmodes; ++m) {
                    if(min_inds_low[stream_idx][m] > inds_low_ptr[m])
                        min_inds_low[stream_idx][m] = inds_low_ptr[m];
                    if(max_inds_high[stream_idx][m] < inds_high_ptr[m])
                        max_inds_high[stream_idx][m] = inds_high_ptr[m];
                }
                /* Determine the range of nonzeros */
                sum_nnz += subtsr_ptr->nnz;
                /* Determine the maximum number of threads for each block */
                if(max_nnz_stream[stream_idx] < subtsr_ptr->nnz)
                    max_nnz_stream[stream_idx] = subtsr_ptr->nnz;

            }   // End loop for block_idx in [0, nblocks-1]
            // printf("max_nnz_stream:\n");
            // spt_DumpArray(max_nnz_stream, stream_idx+1, 0, stdout);


            sptStartTimer(timer);

            /* dev_inds_low */
            result = sptCudaDuplicateMemoryIndirect(&dev_inds_low[stream_idx], inds_low_header, nblocks_count[stream_idx], nmodes, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_Xndims */
            result = sptCudaDuplicateMemoryIndirect(&dev_Xndims[stream_idx], Xndims_header, nblocks_count[stream_idx], nmodes, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_nnz */
            result = sptCudaDuplicateMemory(&dev_nnz[stream_idx], nnz_blk, nblocks * sizeof (size_t), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_nnz_blk_begin */
            result = sptCudaDuplicateMemory(&dev_nnz_blk_begin[stream_idx], nnz_blk_begin, nblocks * sizeof (size_t), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_min_inds_low */
            // printf("min_inds_low[%zu]:\n", stream_idx);
            // spt_DumpArray(min_inds_low[stream_idx], nmodes, 0, stdout);
            result = sptCudaDuplicateMemory(&dev_min_inds_low[stream_idx], min_inds_low[stream_idx], nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* mats_header and lengths, VERY careful of the size and beginning location. */
            for(size_t m = 0; m < nmodes; ++m) {
                mats_header[m] = mats[m]->values + min_inds_low[stream_idx][m] * stride;
                lengths[m] = (max_inds_high[stream_idx][m] - min_inds_low[stream_idx][m]) * stride;
            }
            mats_header[nmodes] = mats[nmodes]->values + min_inds_low[stream_idx][mode] * stride;
            lengths[nmodes] = (max_inds_high[stream_idx][mode] - min_inds_low[stream_idx][mode]) * stride;
            // printf("min_inds_low[%zu]:\n", stream_idx);
            // spt_DumpArray(min_inds_low[stream_idx], nmodes, 0, stdout);
            // printf("lengths[nmodes]: %zu\n", lengths[nmodes]);
            /* dev_mats */
            result = sptCudaDuplicateMemoryIndirect(&dev_mats[stream_idx], mats_header, nmodes+1, lengths, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* Copy back the pointer to dev_mats[stream_idx][nmodes] to the result */
            result = cudaMemcpy(&dev_part_prod[stream_idx], &(dev_mats[stream_idx][nmodes]), sizeof *dev_part_prod, cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            /* dev_part_prod = 0 */
            result = cudaMemset(dev_part_prod[stream_idx], 0, lengths[nmodes] * sizeof (sptScalar));
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            /* Xinds_header */
            // printf("nnz_begin: %zu, sum_nnz: %zu\n", nnz_begin, sum_nnz);
            for(size_t m = 0; m < nmodes; ++m) {
                Xinds_header[m] = tsr->inds[m].data + nnz_begin;
                // printf("Xinds_header[%zu]\n", m);
                // spt_DumpArray(Xinds_header[m], sum_nnz, 0, stdout);
            }
            /* dev_Xinds */
            result = sptCudaDuplicateMemoryIndirect(&dev_Xinds[stream_idx], Xinds_header, nmodes, sum_nnz, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_Xvals */
            result = sptCudaDuplicateMemory(&dev_Xvals[stream_idx], tsr->values.data + nnz_begin, sum_nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            sptStopTimer(timer);
            time_h2d += sptElapsedTime(timer);

            nnz_begin += sum_nnz;

        } // End loop for stream_idx in [0, stream_count-1]

        
        sptStartTimer(timer);
        for(size_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
            size_t nthreadsx, nthreadsy;
            switch(impl_num) {
            case 11:
                nthreadsx = max_nnz_stream[stream_idx];
                nthreadsy = 1;
                break;
            case 15:
            case 16:
            case 17:
                nthreadsy = max_nnz_stream[stream_idx];
                if(R < max_nthreadsy)
                    nthreadsx = R;
                else
                    nthreadsx = max_nthreadsy;
                break;
            }
            dim3 dimBlock (nthreadsx, nthreadsy);
            size_t real_nblocks = nblocks_count[stream_idx];

            switch(nmodes) {
            case 3:
                switch(impl_num) {
                case 11: // Naive
                    printf("spt_MTTKRPKernelBlockNnz3D<%zu, (%u, %u)>\n", real_nblocks, dimBlock.x, dimBlock.y); fflush(stdout);
                    spt_MTTKRPKernelBlockNnz3D<<<real_nblocks, dimBlock>>>(
                        mode,
                        nmodes,
                        dev_nnz[stream_idx],
                        dev_nnz_blk_begin[stream_idx],
                        R,
                        stride,
                        dev_min_inds_low[stream_idx],
                        dev_Xinds[stream_idx],
                        dev_Xvals[stream_idx],
                        dev_mats_order,
                        dev_mats[stream_idx]);
                    break;
                case 15:
                    printf("spt_MTTKRPKernelBlockRankSplitNnz3D<%zu, (%u, %u)>\n", real_nblocks, dimBlock.x, dimBlock.y); fflush(stdout);
                    spt_MTTKRPKernelBlockRankSplitNnz3D<<<real_nblocks, dimBlock>>>(
                        mode,
                        nmodes,
                        dev_nnz[stream_idx],
                        dev_nnz_blk_begin[stream_idx],
                        R,
                        stride,
                        dev_min_inds_low[stream_idx],
                        dev_Xinds[stream_idx],
                        dev_Xvals[stream_idx],
                        dev_mats_order,
                        dev_mats[stream_idx]);
                    break;
                case 16:
                    if(split_grain != 1)
                        printf("Error: wrong impl_num for coarse grain.\n");
                    printf("spt_MTTKRPKernelBlockRankSplitNnz3D_SMCoarse<%zu, (%u, %u)>\n", real_nblocks, dimBlock.x, dimBlock.y); fflush(stdout);
                    spt_MTTKRPKernelBlockRankSplitNnz3D_SMCoarse<<<real_nblocks, dimBlock, smem_size>>>(
                        mode,
                        nmodes,
                        dev_nnz[stream_idx],
                        dev_nnz_blk_begin[stream_idx],
                        R,
                        stride,
                        dev_min_inds_low[stream_idx],
                        dev_inds_low[stream_idx],
                        dev_Xndims[stream_idx],
                        dev_Xinds[stream_idx],
                        dev_Xvals[stream_idx],
                        dev_mats_order,
                        dev_mats[stream_idx]);
                    break;
                case 17:
                    if(split_grain != 3)
                        printf("Error: wrong impl_num for coarse grain.\n");
                    printf("spt_MTTKRPKernelBlockRankSplitNnz3D_SMMedium<%zu, (%u, %u)>\n", real_nblocks, dimBlock.x, dimBlock.y); fflush(stdout);
                    spt_MTTKRPKernelBlockRankSplitNnz3D_SMMedium<<<real_nblocks, dimBlock, smem_size>>>(
                        mode,
                        nmodes,
                        dev_nnz[stream_idx],
                        dev_nnz_blk_begin[stream_idx],
                        R,
                        stride,
                        dev_min_inds_low[stream_idx],
                        dev_inds_low[stream_idx],
                        dev_Xndims[stream_idx],
                        dev_Xinds[stream_idx],
                        dev_Xvals[stream_idx],
                        dev_mats_order,
                        dev_mats[stream_idx]);
                    break;
                }
                break;
            }   // switch nmodes
        } // Loop stream_idx

        for(size_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
            result = cudaDeviceSynchronize();
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }

        sptStopTimer(timer);
        elapsed_time += sptElapsedTime(timer);


        for(size_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
            size_t stream_mode_dim = max_inds_high[stream_idx][mode] - min_inds_low[stream_idx][mode];

            sptStartTimer(timer);
            result = cudaMemcpy(part_prod.values + min_inds_low[stream_idx][mode] * stride, dev_part_prod[stream_idx], stream_mode_dim * stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            sptStopTimer(timer);
            time_d2h += sptElapsedTime(timer);

            // printf("[stream %zu] part_prod:\n", stream_idx);
            // sptDumpMatrix(&part_prod, stdout);

            /* mats[nmodes] += part_prod */
            sptStartTimer(timer);
            #pragma omp parallel for
            for(size_t i = 0; i < stream_mode_dim * stride; ++i) {
                size_t j = i + min_inds_low[stream_idx][mode] * stride;
                mats[nmodes]->values[j] += part_prod.values[j];
            }
            sptStopTimer(timer);
            time_reduce += sptElapsedTime(timer);

        }




        for(size_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
            result = cudaFree(dev_Xndims[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_inds_low[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_mats[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_nnz[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_min_inds_low[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_nnz_blk_begin[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_Xvals[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_Xinds[stream_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }


    } // End loop for sbatch_idx in [0, batched_nstreams-1]


    printf("[CUDA SpTns One MTTKRP (per Stream)]: %lf s\n", elapsed_time);
    printf("\tH2D time: %lf s\n", time_h2d);
    printf("\tD2H time: %lf s\n", time_d2h);
    printf("\treduce time: %lf s\n\n", time_reduce);
    *queue_time = elapsed_time;
    *queue_time_h2d = time_h2d;
    *queue_time_d2h = time_d2h;
    *queue_time_reduce = time_reduce;
    sptFreeTimer(timer);

    delete[] dev_Xvals;
    delete[] dev_Xinds;
    delete[] Xinds_header;
    delete[] dev_mats;
    delete[] mats_header;
    delete[] lengths;
    delete[] dev_nnz;
    delete[] dev_min_inds_low;
    delete[] dev_Xndims;
    delete[] Xndims_header;
    delete[] dev_inds_low;
    delete[] inds_low_header;
    delete[] dev_part_prod;

    result = cudaFree(dev_mats_order);
    spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

    for(size_t i=0; i<max_nstreams; ++i) {
        delete[] min_inds_low[i];
        delete[] max_inds_high[i];
    }
    delete[] min_inds_low;
    delete[] max_inds_high;
    delete[] max_nnz_stream;
    delete[] nblocks_count;
    delete[] nnz_blk;
    delete[] nnz_blk_begin;

    sptFreeMatrix(&part_prod);

    return 0;
}

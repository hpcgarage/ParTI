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

__global__ static void spt_MTTKRPKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch
) {
    const size_t tidx = threadIdx.x;
    const size_t x = blockIdx.x * blockDim.x + tidx;

    size_t const nmats = nmodes - 1;
    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*) dev_mats[nmodes];

    if(x < nnz) {
        size_t times_mat_index = dev_mats_order[0];
        sptScalar * times_mat = dev_mats[times_mat_index];
        size_t * times_inds = Xinds[times_mat_index];
        size_t tmp_i = times_inds[x];
        sptScalar const entry = Xvals[x];
        for(size_t r=0; r<R; ++r) {
            dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
        }

        for(size_t i=1; i<nmats; ++i) {
            times_mat_index = dev_mats_order[i];
            times_mat = dev_mats[times_mat_index];
            times_inds = Xinds[times_mat_index];
            tmp_i = times_inds[x];
            for(size_t r=0; r<R; ++r) {
                dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
            }
        }

    }

    __syncthreads();

    if(x < nnz) {
        size_t const mode_i = mode_ind[x];
        for(size_t r=0; r<R; ++r) {
            atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
        }
    }
    __syncthreads();
}




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
int sptCudaDistributedMTTKRP(
    double *queue_time,
    spt_SplitResult const *splits,
    size_t const queue_size,
    size_t const batch_size,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    int const gpu_map[]
) {
    if(queue_size == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "queue_size == 0");
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

    /* Initialize part_prod matrix, to store partial results of updated mats[mode] */
    sptMatrix part_prod;
    result = sptNewMatrix(&part_prod, mats[mode]->nrows, mats[mode]->ncols);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
  

    /* dev_mats_order[i, m] <= mats_order[i, m] */
    size_t **dev_mats_order = new size_t *[batch_size];
    for(size_t i = 0; i < batch_size; ++i) {
        cudaSetDevice(gpu_map[i]);
        result = sptCudaDuplicateMemory(&dev_mats_order[i], mats_order, nmodes * sizeof *dev_mats_order, cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    }
 

    size_t *dev_nnz = new size_t[batch_size];
    size_t **dev_Xndims = new size_t *[batch_size];
    sptScalar **mats_header = new sptScalar *[nmodes+1];
    size_t * const lengths = new size_t[nmodes+1];
    sptScalar ***dev_mats = new sptScalar **[batch_size];
    size_t **Xinds_header = new size_t *[nmodes];
    size_t ***dev_Xinds = new size_t **[batch_size];
    sptScalar **dev_Xvals = new sptScalar *[batch_size];
    sptScalar **dev_scratch = new sptScalar *[batch_size];

    for(size_t i = 0; i < batch_size; ++i) {
        cudaSetDevice(gpu_map[i]);
        result = cudaMalloc(&dev_Xndims[i], nmodes * sizeof *dev_Xndims);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    }

    size_t batch_count = (queue_size-1)/batch_size + 1;
    // printf("queue_size: %zu; batch_count: %zu\n", queue_size, batch_count);
    const spt_SplitResult * splits_loop = splits;
    double elapsed_time = 0;
    sptTimer timer;
    sptNewTimer(&timer, 0);
    for(size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        /* Index inside batch_size */
        size_t max_kernel_count = batch_idx == batch_count-1 ? queue_size - batch_idx*batch_size : batch_size;
        // printf("max_kernel_count: %zu\n", max_kernel_count);

        size_t kernel_count = 0;
        while(splits_loop != NULL && kernel_count < max_kernel_count) {
            // printf("batch_idx: %zu, kernel_count: %zu\n", batch_idx, kernel_count); fflush(stdout);

            cudaSetDevice(gpu_map[kernel_count]);

            // fprintf(stderr, "[CUDA SpTns SpltMTTKRP] Kernel %zu, device %d\n", kernel_count, gpu_map[kernel_count]);
            // fprintf(stderr, "Input tensor:\n");
            // sptDumpSparseTensor(&splits->tensor, 1, stderr);
            // fprintf(stderr, "\n");


            /* dev_Xndims[kid] <= splits_loop[sid].ndims */
            result = cudaMemcpy(dev_Xndims[kernel_count], splits_loop->tensor.ndims, nmodes * sizeof *dev_Xndims, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_mats[kid] <= mats[i + splits_loop[sid].inds_low, r] */
            for(size_t m = 0; m < nmodes; ++m) {
                mats_header[m] = mats[m]->values + splits_loop->inds_low[m];
                lengths[m] = splits_loop->inds_high[m] - splits_loop->inds_low[m] + 1;
            }
            mats_header[nmodes] = mats[nmodes]->values;
            lengths[nmodes] = sptGetMatrixLength(mats[mode]);
            result = sptCudaDuplicateMemoryIndirect(&dev_mats[kernel_count], mats_header, nmodes+1, lengths, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            dev_nnz[kernel_count] = splits_loop->tensor.nnz;

            /* dev_Xinds[kid, m] <= splits_loop[sid].inds[m] */
            for(size_t m = 0; m < nmodes; ++m) {
                Xinds_header[m] = splits_loop->tensor.inds[m].data;
            }
            result = sptCudaDuplicateMemoryIndirect(&dev_Xinds[kernel_count], Xinds_header, nmodes, splits_loop->tensor.nnz, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_Xvals[kid] <= splits_loop[sid].values */
            result = sptCudaDuplicateMemory(&dev_Xvals[kernel_count], splits_loop->tensor.values.data, splits_loop->tensor.nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_scratch <= zeros(splits_loop[sid].nnz, stride) */
            result = cudaMalloc((void **) &dev_scratch[kernel_count], splits_loop->tensor.nnz * stride * sizeof (sptScalar));
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            ++ kernel_count;
            splits_loop = splits_loop->next;
        }   // End while "splits_loop != NULL && < max_kernel_count"


        sptStartTimer(timer);

        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            size_t nnz = dev_nnz[kernel_idx];

            size_t nthreads = 128;
            size_t nblocks = (nnz + nthreads -1) / nthreads;

            cudaSetDevice(gpu_map[kernel_idx]);

            spt_MTTKRPKernel<<<nblocks, nthreads>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims[kernel_idx],
                dev_Xinds[kernel_idx],
                dev_Xvals[kernel_idx],
                dev_mats_order[kernel_idx],
                dev_mats[kernel_idx],
                dev_scratch[kernel_idx]
            );
        }   // End executing all "kernel_count" kernels

        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            cudaSetDevice(gpu_map[kernel_idx]);
            result = cudaDeviceSynchronize();
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }

        sptStopTimer(timer);
        elapsed_time += sptElapsedTime(timer);

        // TODO!
        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            cudaSetDevice(gpu_map[kernel_idx]);

            /* Copy back the pointer to the result */
            sptScalar *dev_part_prod;
            cudaMemcpy(&dev_part_prod, dev_mats[kernel_idx] + nmodes, sizeof dev_part_prod, cudaMemcpyDeviceToHost);

            /* dev_part_prod => part_prod */
            result = cudaMemcpy(part_prod.values, dev_part_prod, sptGetMatrixLength(&part_prod) * sizeof (sptScalar), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            // fprintf(stderr, "[CUDA SpTns SpltMTTKRP] Kernel %zu, device %d\n", kernel_idx, gpu_map[kernel_idx]);
            // fprintf(stderr, "Output matrix:\n");
            // sptDumpMatrix(&part_prod, stderr);
            // fprintf(stderr, "\n");


            /* mats[nmodes] += part_prod */
            for(size_t i = 0; i < sptGetMatrixLength(&part_prod); ++i) {
                mats[nmodes]->values[i] += part_prod.values[i];
            }

            /* dev_part_prod = 0 */
            /* Why mix inputs & outputs together inside these deep device pointers?! */
            result = cudaMemset(dev_part_prod, 0, sptGetMatrixLength(&part_prod) * sizeof (sptScalar));
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }

        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            cudaSetDevice(gpu_map[kernel_idx]);
            result = cudaFree(dev_mats[kernel_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_scratch[kernel_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_Xvals[kernel_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_Xinds[kernel_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }
    }   // End queue_size

    printf("[CUDA SpTns Dist MTTKRP (per Queue)]: %lf s\n\n", elapsed_time);
    *queue_time = elapsed_time;
    sptFreeTimer(timer);

    delete[] dev_scratch;
    delete[] dev_Xvals;
    delete[] dev_Xinds;
    delete[] Xinds_header;
    delete[] mats_header;
    delete[] lengths;
    delete[] dev_mats;

    for(size_t i = 0; i < batch_size; ++i) {
        cudaSetDevice(gpu_map[i]);
        result = cudaFree(dev_mats_order[i]);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
        result = cudaFree(dev_Xndims[i]);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    }
    delete[] dev_mats_order;
    delete[] dev_Xndims;

    sptFreeMatrix(&part_prod);


    // fprintf(stderr, "[CUDA SpTns SpltMTTKRP] Result matrix\n");
    // sptDumpMatrix(mats[nmodes], stderr);
    // fprintf(stderr, "\n");


    return 0;
}

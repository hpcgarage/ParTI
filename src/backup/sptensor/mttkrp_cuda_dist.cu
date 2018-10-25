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
 * Do NOT initialize result matrix, mats[nmodes], inside.
 */
int sptCudaDistributedMTTKRP(
    double *queue_time,
    int const split_grain,
    spt_SplitResult const *splits,
    size_t const queue_size,    // Acurate size!
    size_t const batch_size,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    int const gpu_map[]) 
{
    if(queue_size == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "queue_size == 0");
    }
    if(batch_size == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "batch_size == 0");
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

    /* Initialize part_prod matrix, to store partial results of updated mats[nmodes] */
    sptMatrix part_prod;
    result = sptNewMatrix(&part_prod, mats[mode]->nrows, mats[mode]->ncols);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
  

    /* accurate_Xndims: 1st cpu. The accurate ndims range for each subtsr */
    size_t * accurate_Xndims = new size_t[nmodes];
    /* dev_inds_low: 1st cpu, 2nd gpu. One copy of inds_low per gpu */
    size_t **dev_inds_low = new size_t *[batch_size];
    /* dev_mats_order: 1st cpu, 2nd gpu. One copy of mats_order per gpu */
    size_t **dev_mats_order = new size_t *[batch_size];
    /* dev_Xndims: 1st cpu, 2nd gpu. One copy of subtsr's Xndims per gpu */
    size_t **dev_Xndims = new size_t *[batch_size];
    /* dev_nnz: 1st cpu. One copy of subtsr's nnz per gpu */
    size_t *dev_nnz = new size_t[batch_size];
    /* mats_header: 1st cpu, 2nd cpu (ghost pointers) */
    sptScalar **mats_header = new sptScalar *[nmodes+1];
    /* lengths: 1st cpu, store the lengths of mats */
    size_t * const lengths = new size_t[nmodes+1];
    /* dev_mats: 1st cpu, 2nd gpu, 3rd gpu. One copy of subtsr's corresponding mats per gpu */
    sptScalar ***dev_mats = new sptScalar **[batch_size]; 
    /* dev_part_prod: 1st gpu, 2nd gpu, the pointer to dev_mats[gpu_idx][nmodes] */
    sptScalar **dev_part_prod = new sptScalar*[batch_size];
    /* Xinds_header: 1st cpu, 2nd cpu (ghost pointers) */
    size_t **Xinds_header = new size_t *[nmodes];
    /* dev_Xinds: 1st cpu, 2nd gpu, 3rd gpu. One copy of subtsr's inds per gpu */
    size_t ***dev_Xinds = new size_t **[batch_size];
    /* dev_Xvals: 1st cpu, 2nd gpu. One copy of subtsr's vals per gpu */
    sptScalar **dev_Xvals = new sptScalar *[batch_size];
    /* dev_scratch: 1st cpu, 2nd gpu. One copy of scratch per gpu */
    sptScalar **dev_scratch = new sptScalar *[batch_size];

    for(size_t i = 0; i < batch_size; ++i) {
        cudaSetDevice(gpu_map[i]);
        result = sptCudaDuplicateMemory(&dev_mats_order[i], mats_order, nmodes * sizeof *dev_mats_order, cudaMemcpyHostToDevice);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

        /* Only malloc once, because of equal sizes */
        result = cudaMalloc(&dev_Xndims[i], nmodes * sizeof *dev_Xndims);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
        result = cudaMalloc(&dev_inds_low[i], nmodes * sizeof *dev_inds_low);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    }


    size_t batch_count = (queue_size-1)/batch_size + 1;
    // printf("queue_size: %zu; batch_count: %zu\n", queue_size, batch_count);
    double elapsed_time = 0;
    sptTimer timer;
    sptNewTimer(&timer, 0);

    for(size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        /* Index inside batch_size */
        size_t gpu_count = batch_idx == batch_count-1 ? queue_size - batch_idx*batch_size : batch_size;
        // printf("gpu_count: %zu\n", gpu_count);

        for(size_t gpu_idx = 0; gpu_idx < gpu_count; ++gpu_idx) {
            size_t queue_idx = batch_idx * batch_count + gpu_idx;     
            sptAssert(queue_idx < queue_size);       
            sptAssert(&(splits[queue_idx].tensor) != NULL);
            printf("queue_idx: %zu, batch_idx: %zu, gpu_idx: %zu\n", queue_idx, batch_idx, gpu_count); fflush(stdout);
            // spt_SparseTensorDumpAllSplits(&splits[queue_idx], 1, stdout);
            cudaSetDevice(gpu_map[gpu_idx]);

            sptSparseTensor const * subtsr_ptr = &(splits[queue_idx].tensor);
            size_t * inds_low_ptr = splits[queue_idx].inds_low;
            size_t * inds_high_ptr = splits[queue_idx].inds_high;
            for(size_t i=0; i<nmodes; ++i)
                accurate_Xndims[i] = inds_high_ptr[i] - inds_low_ptr[i];

            // fprintf(stdout, "[CUDA SpTns SpltMTTKRP] Kernel %zu, device %d\n", gpu_idx, gpu_map[gpu_idx]);
            // fprintf(stdout, "Input tensor:\n");
            // sptDumpSparseTensor(subtsr_ptr, 0, stdout);
            // fprintf(stdout, "\n");

            /* dev_Xndims */
            result = cudaMemcpy(dev_Xndims[gpu_idx], accurate_Xndims, nmodes * sizeof *dev_Xndims, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
            /* dev_inds_low */
            result = cudaMemcpy(dev_inds_low[gpu_idx], inds_low_ptr, nmodes * sizeof *dev_inds_low, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_nnz */
            dev_nnz[gpu_idx] = subtsr_ptr->nnz;

            /* mats_header and lengths, VERY careful of the size and beginning location. */
            for(size_t m = 0; m < nmodes; ++m) {
                mats_header[m] = mats[m]->values + inds_low_ptr[m] * stride;
                lengths[m] = (inds_high_ptr[m] - inds_low_ptr[m]) * stride;
            }
            mats_header[nmodes] = mats[nmodes]->values + inds_low_ptr[mode] * stride;
            lengths[nmodes] = (inds_high_ptr[mode] - inds_low_ptr[mode]) * stride;
            /* dev_mats */
            result = sptCudaDuplicateMemoryIndirect(&dev_mats[gpu_idx], mats_header, nmodes+1, lengths, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* Copy back the pointer to dev_mats[gpu_idx][nmodes] to the result */
            result = cudaMemcpy(&dev_part_prod[gpu_idx], dev_mats[gpu_idx] + nmodes, sizeof *dev_part_prod, cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            /* dev_part_prod = 0 */
            /* Why mix inputs & outputs together inside these deep device pointers?! */
            result = cudaMemset(dev_part_prod[gpu_idx], 0, accurate_Xndims[mode] * stride * sizeof (sptScalar));
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            /* Xinds_header */
            for(size_t m = 0; m < nmodes; ++m) {
                Xinds_header[m] = subtsr_ptr->inds[m].data;
            }
            /* dev_Xinds */
            result = sptCudaDuplicateMemoryIndirect(&dev_Xinds[gpu_idx], Xinds_header, nmodes, subtsr_ptr->nnz, cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_Xvals */
            result = sptCudaDuplicateMemory(&dev_Xvals[gpu_idx], subtsr_ptr->values.data, subtsr_ptr->nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            /* dev_scratch */
            result = cudaMalloc((void **) &dev_scratch[gpu_idx], subtsr_ptr->nnz * stride * sizeof (sptScalar));
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

        }   // End for loop: gpu_idx

        sptStartTimer(timer);

        for(size_t gpu_idx = 0; gpu_idx < gpu_count; ++gpu_idx) {
            size_t nthreads = 128;
            size_t nblocks = (dev_nnz[gpu_idx] + nthreads -1) / nthreads;

            cudaSetDevice(gpu_map[gpu_idx]);

            spt_MTTKRPKernelScratchDist<<<nblocks, nthreads>>>(
                mode,
                nmodes,
                dev_nnz[gpu_idx],
                R,
                stride,
                dev_Xndims[gpu_idx],
                dev_inds_low[gpu_idx],
                dev_Xinds[gpu_idx],
                dev_Xvals[gpu_idx],
                dev_mats_order[gpu_idx],
                dev_mats[gpu_idx],
                dev_scratch[gpu_idx]
            );
        }   // End executing all "gpu_count" kernels


        for(size_t gpu_idx = 0; gpu_idx < gpu_count; ++gpu_idx) {
            cudaSetDevice(gpu_map[gpu_idx]);
            result = cudaDeviceSynchronize();
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }

        sptStopTimer(timer);
        elapsed_time += sptElapsedTime(timer);


        for(size_t gpu_idx = 0; gpu_idx < gpu_count; ++gpu_idx) {
            size_t queue_idx = batch_idx * batch_count + gpu_idx;
            cudaSetDevice(gpu_map[gpu_idx]);

            size_t * inds_low_ptr = splits[queue_idx].inds_low;
            size_t * inds_high_ptr = splits[queue_idx].inds_high;
            for(size_t i=0; i<nmodes; ++i)
                accurate_Xndims[i] = inds_high_ptr[i] - inds_low_ptr[i];

            // spt_DumpArray(inds_low_ptr, nmodes, 0, stdout);
            // spt_DumpArray(accurate_Xndims, nmodes, 0, stdout);
            result = cudaMemcpy(part_prod.values + inds_low_ptr[mode] * stride, dev_part_prod, accurate_Xndims[mode] * stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

            // fprintf(stdout, "[CUDA SpTns SpltMTTKRP] Kernel %zu, device %d\n", gpu_idx, gpu_map[gpu_idx]);
            // fprintf(stdout, "part_prod:\n");
            // sptDumpMatrix(&part_prod, stdout);
            // fprintf(stdout, "\n");

            /* mats[nmodes] += part_prod */
            #pragma omp parallel for
            for(size_t i = 0; i < accurate_Xndims[mode] * stride; ++i) {
                size_t j = i + inds_low_ptr[mode] * stride;
                mats[nmodes]->values[j] += part_prod.values[j];
            }

            // fprintf(stdout, "[CUDA SpTns SpltMTTKRP] Kernel %zu, device %d\n", gpu_idx, gpu_map[gpu_idx]);
            // fprintf(stdout, "Output matrix:\n");
            // sptDumpMatrix(mats[nmodes], stdout);
            // fprintf(stdout, "\n");
        }

        for(size_t gpu_idx = 0; gpu_idx < gpu_count; ++gpu_idx) {
            cudaSetDevice(gpu_map[gpu_idx]);
            result = cudaFree(dev_mats[gpu_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_scratch[gpu_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_Xvals[gpu_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
            result = cudaFree(dev_Xinds[gpu_idx]);
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }
    }   // End nbatches

    printf("[CUDA SpTns Dist MTTKRP (per Queue)]: %lf s\n\n", elapsed_time);
    *queue_time = elapsed_time;
    sptFreeTimer(timer);

    delete[] dev_scratch;
    delete[] dev_Xvals;
    delete[] dev_Xinds;
    delete[] Xinds_header;
    delete[] mats_header;
    delete[] lengths;
    delete[] dev_nnz;
    delete[] dev_mats;
    delete[] dev_part_prod;

    for(size_t i = 0; i < batch_size; ++i) {
        cudaSetDevice(gpu_map[i]);
        result = cudaFree(dev_mats_order[i]);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
        result = cudaFree(dev_Xndims[i]);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
        result = cudaFree(dev_inds_low[i]);
        spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    }
    delete[] dev_mats_order;
    delete[] dev_Xndims;
    delete[] dev_inds_low;
    delete[] accurate_Xndims;

    sptFreeMatrix(&part_prod);


    // fprintf(stderr, "[CUDA SpTns SpltMTTKRP] Result matrix\n");
    // sptDumpMatrix(mats[nmodes], stderr);
    // fprintf(stderr, "\n");


    return 0;
}

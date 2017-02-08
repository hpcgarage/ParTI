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


/* A thread block compute a sub-tensor */
__global__ static void spt_MTTKRPKernelSM(
  const size_t mode,
  const size_t nmodes,
  const size_t nnz,
  const size_t R,
  const size_t stride,
  const size_t * Xndims,
  const size_t nsplits,
  size_t ** const split_Xndims,
  size_t * const split_nnz,
  size_t *** const split_Xinds,
  sptScalar ** const split_Xvals,
  const size_t * dev_mats_order,
  sptScalar ** dev_mats,
  size_t block_offset) 
{
  const size_t tidx = threadIdx.x;
  const size_t bidx = blockIdx.x + block_offset;

#if 1
  extern __shared__ sptScalar sm_pool[];
  size_t const nmats = nmodes - 1;
  size_t * split_low = split_Xndims[bidx];
  size_t const block_nnz = split_nnz[bidx];

  /* data in shared memory */
  size_t * sm_dev_mats_order = (size_t *)sm_pool; // dev_mats_order
  size_t * sm_split_low = sm_dev_mats_order + nmats; // split_low
  size_t * sm_split_high = sm_split_low + nmodes; // split_high

  if(nmats + 2 * nmodes < blockDim.x) {
    if(tidx < nmats) {
      sm_dev_mats_order[tidx] = dev_mats_order[tidx];
    } else if (tidx >= nmats && tidx < nmats + 2 * nmodes) {
      sm_split_low[tidx - nmats] = split_low[tidx - nmats];
    }
  } else {
    if(tidx < nmats) {
      sm_dev_mats_order[tidx] = dev_mats_order[tidx];
    }
    if (tidx < 2 * nmodes) {
      sm_split_low[tidx] = split_low[tidx];
    }
  }
  __syncthreads();

  sptScalar * sm_dev_scratch = (sptScalar *)(sm_split_high + nmodes);  // mode matrix
  sptScalar * sm_times_mat = sm_dev_scratch + block_nnz * R; // Timing matrix, this replace need to be reused.

  size_t times_mat_mode = sm_dev_mats_order[0];
  sptScalar * times_mat = dev_mats[times_mat_mode];
  size_t times_mat_nrows = sm_split_high[times_mat_mode] - sm_split_low[times_mat_mode];
  if(tidx < times_mat_nrows) {
    for(size_t r=0; r<R; ++r)
      sm_times_mat[tidx * stride + r] = times_mat[(tidx+sm_split_low[times_mat_mode]) * stride + r];
  }
  __syncthreads();
  size_t * times_inds = split_Xinds[bidx][times_mat_mode];
  size_t index;
  sptScalar entry;
  if(tidx < block_nnz) {
    index = times_inds[tidx] - sm_split_low[times_mat_mode];
    entry = split_Xvals[bidx][tidx];
    for(size_t r=0; r<R; ++r) {
      sm_dev_scratch[tidx * stride + r] = entry * sm_times_mat[index * stride + r];
    }
  }
  __syncthreads();

  for(size_t i=1; i<nmats; ++i) {
    times_mat_mode = sm_dev_mats_order[i];
    times_mat = dev_mats[times_mat_mode];
    times_mat_nrows = sm_split_high[times_mat_mode] - sm_split_low[times_mat_mode];
    if(tidx < times_mat_nrows) {
      for(size_t r=0; r<R; ++r)
        sm_times_mat[tidx * stride + r] = times_mat[(tidx+sm_split_low[times_mat_mode]) * stride + r];
    }
    __syncthreads();
    times_inds = split_Xinds[bidx][times_mat_mode];
    if(tidx < block_nnz) {
      index = times_inds[tidx] - sm_split_low[times_mat_mode];
      for(size_t r=0; r<R; ++r) {
        sm_dev_scratch[tidx * stride + r] *= sm_times_mat[index * stride + r];
      }
    }
    __syncthreads();
  }

  size_t const * const mode_ind = split_Xinds[bidx][mode];
  sptScalar * const mvals = dev_mats[nmodes];
  size_t mmat_nrows = sm_split_high[mode] - sm_split_low[mode];
  if(tidx < mmat_nrows) {
    for(size_t r=0; r<R; ++r)
      sm_times_mat[tidx * stride + r] = mvals[(tidx+sm_split_low[mode]) * stride + r];
  }
  __syncthreads();

  
  if(tidx < block_nnz) {
    size_t const mode_i = mode_ind[tidx] - sm_split_low[mode];
    for(size_t r=0; r<R; ++r) {
      atomicAdd(&(sm_times_mat[mode_i * stride + r]), sm_dev_scratch[tidx * stride + r]);
    }
  }
  __syncthreads();


  if(tidx < mmat_nrows) {
    for(size_t r=0; r<R; ++r)
      atomicAdd(&(mvals[(tidx+sm_split_low[mode]) * stride + r]), sm_times_mat[tidx * stride + r]);
  }
  __syncthreads();
#endif

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
int sptCudaMTTKRPSM(sptSparseTensor const * const X,
  sptMatrix ** const mats,     // mats[nmodes] as temporary space.
  sptSizeVector const * const mats_order,    // Correspond to the mode order of X.
  size_t const mode) {

  size_t const memory_size = 49152; // Shared memory size
  size_t const nmodes = X->nmodes;
  size_t const nnz = X->nnz;
  size_t const * const ndims = X->ndims;
  size_t const R = mats[mode]->ncols;
  size_t const stride = mats[mode]->stride;
  size_t const nmats = nmodes - 1;
  int result;

  /* Check the mats. */
  for(size_t i=0; i<nmodes; ++i) {
      if(mats[i]->ncols != mats[nmodes]->ncols) {
          spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
      }
      if(mats[i]->nrows != ndims[i]) {
          spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
      }
  }


  size_t const memory_step = (size_t) (0.9 * memory_size/(sizeof(size_t)*R*2));
  printf("memory_step: %lu\n", memory_step);
  size_t *steps = (size_t*)malloc(nmodes * sizeof (size_t));
  for(size_t i=0; i<nmodes; ++i)
    steps[i] = memory_step;
  spt_SplitResult *splits;
  size_t nsplits;
  sptAssert(spt_SparseTensorGetAllSplits(&splits, &nsplits, X, steps, NULL, 1) == 0);
  spt_SparseTensorDumpAllSplits(splits, nsplits, stdout);


  /* Transfer tensor and matrices */
  size_t * Xndims = NULL;
  result = sptCudaDuplicateMemory(&Xndims, ndims, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  size_t * dev_mats_order = NULL;
  result = sptCudaDuplicateMemory(&dev_mats_order,  mats_order->data, nmats * sizeof(size_t), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  sptScalar ** tmp_mats = NULL;
  tmp_mats = (sptScalar **)malloc((nmodes+1) * sizeof(sptScalar*));
  for(size_t i=0; i<nmodes+1; ++i) {
    result = sptCudaDuplicateMemory(&(tmp_mats[i]), mats[i]->values, 
      mats[i]->nrows * mats[i]->stride * sizeof(sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  result = cudaMemset(tmp_mats[nmodes], 0, mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  sptScalar ** dev_mats = NULL;   // array of pointer to device memory
  result = sptCudaDuplicateMemory(&dev_mats, tmp_mats, (nmodes+1) * sizeof (sptScalar*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


  size_t ** tmp_Xndims = NULL;
  tmp_Xndims = (size_t **)malloc(nsplits * sizeof(size_t*));
  for(size_t i=0; i<nsplits; ++i) {
    result = sptCudaDuplicateMemory(&(tmp_Xndims[i]), splits[i].inds_low, 
      2 * nmodes * sizeof(size_t), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  size_t ** split_Xndims = NULL;   // array of pointer to device memory
  result = cudaMalloc((void***)&split_Xndims, nsplits * sizeof(size_t*));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(split_Xndims, tmp_Xndims, nsplits * sizeof (size_t*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


  size_t * tmp_nnz = (size_t *)malloc(nsplits * sizeof(size_t));
  for(size_t i=0; i<nsplits; ++i) {
    tmp_nnz[i] = splits[i].tensor.nnz;
  }
  size_t * split_nnz = NULL;
  result = sptCudaDuplicateMemory(&split_nnz, tmp_nnz, nsplits * sizeof(size_t), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  free(tmp_nnz);


  size_t *** tmp1_split_Xinds = (size_t ***)malloc(nsplits * sizeof(size_t**));
  for(size_t i=0; i<nsplits; ++i) {
    tmp1_split_Xinds[i] = (size_t **)malloc(nmodes * sizeof(size_t*));
    for(size_t m=0; m<nmodes; ++m) {
      result = sptCudaDuplicateMemory(&(tmp1_split_Xinds[i][m]), splits[i].tensor.inds[m].data, 
        splits[i].tensor.nnz * sizeof(size_t), cudaMemcpyHostToDevice);
      spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    }  
  }
  size_t *** tmp2_split_Xinds = (size_t ***)malloc(nsplits * sizeof(size_t**));
  for(size_t i=0; i<nsplits; ++i) {
    result = cudaMalloc((void***)&(tmp2_split_Xinds[i]), nmodes * sizeof(size_t*));
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaMemcpy(tmp2_split_Xinds[i], tmp1_split_Xinds[i], nmodes * sizeof(size_t*), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  size_t *** split_Xinds = NULL;   // array of pointer to device memory
  result = cudaMalloc((void ****) &split_Xinds, nsplits * sizeof(size_t**));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(split_Xinds, tmp2_split_Xinds, nsplits * sizeof (size_t**), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


  sptScalar ** split_Xvals = NULL;
  sptScalar ** tmp_split_Xvals = (sptScalar **)malloc(nsplits * sizeof(sptScalar*));
  for(size_t i=0; i<nsplits; ++i) {
    result = sptCudaDuplicateMemory(&(tmp_split_Xvals[i]), splits[i].tensor.values.data, 
      splits[i].tensor.nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  result = cudaMalloc((void ***) &split_Xvals, nsplits * sizeof(sptScalar*));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(split_Xvals, tmp_split_Xvals, nsplits * sizeof (sptScalar*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


  const size_t nthreads = 128;
  const size_t max_nblocks = 32768;
  printf("nsplits: %lu, nthreads: %lu\n", nsplits, nthreads);
  size_t max_block_nnz = 0;
  for(size_t i=0; i<nsplits; ++i) {
    if(max_block_nnz < splits[i].tensor.nnz)
      max_block_nnz = splits[i].tensor.nnz;
  }
  size_t allocate_sm_size = (nmats + 2 * nmodes) * sizeof(size_t) + 2 * max_block_nnz * R * sizeof(sptScalar);
  printf("max_block_nnz: %lu\n", max_block_nnz);
  printf("allocate_sm_size: %lu, given shared memory size: %lu\n", allocate_sm_size, memory_size);
  sptAssert (allocate_sm_size < memory_size);


  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  for(size_t block_offset = 0; block_offset < nsplits; block_offset += max_nblocks) {
    size_t nblocks = nsplits - block_offset;
    if(nblocks > max_nblocks) {
        nblocks = max_nblocks;
    }
    spt_MTTKRPKernelSM<<<nblocks, nthreads, memory_size>>>(
        mode,
        nmodes,
        nnz,
        R,
        stride,
        Xndims,
        nsplits,
        split_Xndims,
        split_nnz,
        split_Xinds,
        split_Xvals,
        dev_mats_order,
        dev_mats,
        block_offset
        );
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }


  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP");
  sptFreeTimer(timer);


  result = cudaMemcpy(mats[nmodes]->values, tmp_mats[nmodes], mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP copy back");


  result = cudaFree(Xndims);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(dev_mats_order);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  for(size_t i=0; i<nsplits; ++i) {
    result = cudaFree(tmp_split_Xvals[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  free(tmp_split_Xvals);
  result = cudaFree(split_Xvals);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  
  for(size_t i=0; i<nsplits; ++i) {
    for(size_t m=0; m<nmodes; ++m) {
      result = cudaFree(tmp1_split_Xinds[i][m]);
      spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    }
    free(tmp1_split_Xinds[i]);
  }
  free(tmp1_split_Xinds);
  for(size_t i=0; i<nsplits; ++i) {
    result = cudaFree(tmp2_split_Xinds[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  free(tmp2_split_Xinds);
  result = cudaFree(split_Xinds);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  for(size_t i=0; i<nmodes+1; ++i) {
    result = cudaFree(tmp_mats[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  free(tmp_mats);
  result = cudaFree(dev_mats);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  spt_SparseTensorFreeAllSplits(splits);


  return 0;
}



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


__device__ void lock(int* mutex) {
  /* compare mutex to 0.
     when it equals 0, set it to 1
     we will break out of the loop after mutex gets set to  */
    while (atomicCAS(mutex, 0, 1) != 0) {
    /* do nothing */
    }
}


__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
}


/* impl_num = 01 */
__global__ void spt_MTTKRPKernelNnz3D(
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
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    size_t const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    if(x < nnz) {
      size_t const mode_i = mode_ind[x];
      size_t times_mat_index = dev_mats_order[1];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[x];
      sptScalar const entry = Xvals[x];
      size_t times_mat_index_2 = dev_mats_order[2];
      sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
      size_t * times_inds_2 = Xinds[times_mat_index_2];
      size_t tmp_i_2 = times_inds_2[x];
      sptScalar tmp_val = 0;
      for(size_t r=0; r<R; ++r) {
        tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
        atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
      }

    }
   __syncthreads();

}


/* impl_num = 02 */
__global__ void spt_MTTKRPKernelNnzRank3D(
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
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;
    // printf("x: %lu, tidx: %lu, tidy: %lu\n", x, tidx, tidy);

    size_t const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    if(x < nnz && tidy < R) {
      size_t const mode_i = mode_ind[x];
      size_t times_mat_index = dev_mats_order[1];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[x];
      sptScalar const entry = Xvals[x];
      size_t times_mat_index_2 = dev_mats_order[2];
      sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
      size_t * times_inds_2 = Xinds[times_mat_index_2];
      size_t tmp_i_2 = times_inds_2[x];
      sptScalar tmp_val = 0;

      tmp_val = entry * times_mat[tmp_i * stride + tidy] * times_mat_2[tmp_i_2 * stride + tidy];
      // printf("x: %lu, tidy: %lu, tmp_val: %lf\n", x, tidy, tmp_val);
      atomicAdd(&(mvals[mode_i * stride + tidy]), tmp_val);      
    }
   __syncthreads();

}



/* impl_num = 04 */
__global__ void spt_MTTKRPKernelNnzRankExchangexy3D(
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
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    const size_t x = (blockIdx.x + block_offset) * blockDim.y + tidy;

    size_t const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    if(x < nnz && tidx < R) {
      size_t const mode_i = mode_ind[x];
      size_t times_mat_index = dev_mats_order[1];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[x];
      sptScalar const entry = Xvals[x];
      size_t times_mat_index_2 = dev_mats_order[2];
      sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
      size_t * times_inds_2 = Xinds[times_mat_index_2];
      size_t tmp_i_2 = times_inds_2[x];
      sptScalar tmp_val = 0;

      tmp_val = entry * times_mat[tmp_i * stride + tidx] * times_mat_2[tmp_i_2 * stride + tidx];
      atomicAdd(&(mvals[mode_i * stride + tidx]), tmp_val);      
    }
   __syncthreads();

}



/* impl_num = 03 */
__global__ void spt_MTTKRPKernelNnzRankSplit3D(
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
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;
    const size_t rank_size = R / blockDim.y;  // R is dividable to blockDim.y

    size_t const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    if(x < nnz && tidy * rank_size < R) {
      size_t const mode_i = mode_ind[x];
      size_t times_mat_index = dev_mats_order[1];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[x];
      sptScalar const entry = Xvals[x];
      size_t times_mat_index_2 = dev_mats_order[2];
      sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
      size_t * times_inds_2 = Xinds[times_mat_index_2];
      size_t tmp_i_2 = times_inds_2[x];
      sptScalar tmp_val = 0;

      for(size_t r=tidy*rank_size; r<(tidy+1)*rank_size; ++r) {
        tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
        atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
      }

    }
   __syncthreads();

}


/* impl_num = 09 */
__global__ void spt_MTTKRPKernelScratch(
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
    sptScalar * dev_scratch,
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    size_t const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    if(x < nnz) {
      size_t times_mat_index = dev_mats_order[1];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[x];
      sptScalar const entry = Xvals[x];
      for(size_t r=0; r<R; ++r) {
        dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
      }

      for(size_t i=2; i<nmodes; ++i) {
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
      // printf("x: %lu, mode_i: %lu\n", x, mode_i);
      for(size_t r=0; r<R; ++r) {
        atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
      }
    }
   __syncthreads();

}


/* impl_num = 11 */
__global__ void spt_MTTKRPKernelBlockNnz3D(
    const size_t mode,
    const size_t nmodes,
    const size_t * nnz,
    const size_t * dev_nnz_blk_begin,
    const size_t R,
    const size_t stride,
    size_t * const inds_low_allblocks,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats
) {
    const size_t tidx = threadIdx.x;
    const size_t bidx = blockIdx.x;

    /* block range */
    const size_t nnz_blk = nnz[bidx];
    const size_t nnz_blk_begin = dev_nnz_blk_begin[bidx];
    if(tidx == 0)
        printf("bidx: %lu, nnz_blk: %lu, nnz_blk_begin: %lu\n", bidx, nnz_blk, nnz_blk_begin);

    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    if(tidx < nnz_blk) {
      size_t const mode_i = mode_ind[tidx + nnz_blk_begin] - inds_low_allblocks[mode];    // local base
      // printf("[tidx: %lu, bidx: %lu] global: %lu, mode_i: %lu\n", tidx, bidx, mode_ind[tidx + nnz_blk_begin], mode_i);
      size_t times_mat_index = dev_mats_order[1];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[tidx + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
      sptScalar const entry = Xvals[tidx + nnz_blk_begin];
      size_t times_mat_index_2 = dev_mats_order[2];
      sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
      size_t * times_inds_2 = Xinds[times_mat_index_2];
      size_t tmp_i_2 = times_inds_2[tidx + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
      sptScalar tmp_val = 0;
      printf("[tidx: %lu, bidx: %lu] entry: %f, 1st: %f, 2nd: %f\n", tidx, bidx, entry, times_mat[tmp_i * stride + 0], times_mat_2[tmp_i_2 * stride + 0]);
      for(size_t r=0; r<R; ++r) {
        tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
        atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
      }

    }
   __syncthreads();

}



/* impl_num = 29 */
__global__ void spt_MTTKRPKernelScratchDist(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    const size_t * inds_low,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch
) {
    const size_t tidx = threadIdx.x;
    const size_t x = blockIdx.x * blockDim.x + tidx;

    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = dev_mats[nmodes];

    if(x == 0) {
        printf("mvals:\n");
        for(size_t i=0; i<Xndims[mode]; ++i) {
            printf("%lf\n", mvals[i * stride]);
        }
        printf("mvals end\n");
    }

    if(x < nnz) {
        size_t times_mat_index = dev_mats_order[1];
        sptScalar * times_mat = dev_mats[times_mat_index];
        size_t * times_inds = Xinds[times_mat_index];
        size_t tmp_i = times_inds[x] - inds_low[times_mat_index];
        sptScalar const entry = Xvals[x];
        for(size_t r=0; r<R; ++r) {
            dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
        }

        for(size_t i=2; i<nmodes; ++i) {
            times_mat_index = dev_mats_order[i];
            times_mat = dev_mats[times_mat_index];
            times_inds = Xinds[times_mat_index];
            tmp_i = times_inds[x] - inds_low[times_mat_index];
            for(size_t r=0; r<R; ++r) {
                dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
            }
        }

    }

    __syncthreads();

    if(x < nnz) {
        size_t const mode_i = mode_ind[x] - inds_low[mode];
        // printf("x: %lu, mode_ind[x]: %lu, mode_i: %lu\n", x, mode_ind[x], mode_i);
        for(size_t r=0; r<R; ++r) {
            atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
        }
    }
    __syncthreads();

    // if(x == 0) {
    //     printf("inds_low[mode]: %lu, Xndims[mode]: %lu\n", inds_low[mode], Xndims[mode]);
    //     printf("nnz: %lu\n", nnz);;
    //     printf("mvals:\n");
    //     for(size_t i=0; i<Xndims[mode]; ++i) {
    //         printf("%lf\n", mvals[i * stride]);
    //     }
    //     printf("mvals end\n");
    // }
    
}

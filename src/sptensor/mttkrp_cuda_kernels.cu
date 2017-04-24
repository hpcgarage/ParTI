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
    size_t block_offset) 
{
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
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    size_t const * const mode_ind = Xinds[mode];
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
      atomicAdd(&(mvals[mode_i * stride + tidy]), tmp_val);      
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
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;
    const size_t rank_size = R / blockDim.y;  // R is dividable to blockDim.y

    size_t const * const mode_ind = Xinds[mode];
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


/* impl_num = 04 */
__global__ void spt_MTTKRPKernelRankNnz3D(
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
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    const size_t x = (blockIdx.x + block_offset) * blockDim.y + tidy;

    size_t const * const mode_ind = Xinds[mode];
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


/* impl_num = 05 */
__global__ void spt_MTTKRPKernelRankSplitNnz3D(
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
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    const size_t x = (blockIdx.x + block_offset) * blockDim.y + tidy;
    const size_t num_loops = R / blockDim.x;
    const size_t rest_loop = R - num_loops * blockDim.x;


    size_t const * const mode_ind = Xinds[mode];
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
        size_t r;

        for(size_t l=0; l<num_loops; ++l) {
            r = tidx + l * blockDim.x;
            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            __syncthreads();
        }

        if(rest_loop > 0 && tidx < rest_loop) {
            r = tidx + num_loops * blockDim.x;
            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            __syncthreads();
        }
    }
   

}



/* impl_num = 06 */
__global__ void spt_MTTKRPKernelRankSplitNnzRB3D(
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
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    const size_t x = (blockIdx.x + block_offset) * blockDim.y + tidy;
    const size_t num_loops = R / blockDim.x;
    const size_t rest_loop = R - num_loops * blockDim.x;
    size_t r;

    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];
    size_t times_mat_index = dev_mats_order[1];
    sptScalar * times_mat = dev_mats[times_mat_index];
    size_t * times_inds = Xinds[times_mat_index];
    size_t times_mat_index_2 = dev_mats_order[2];
    sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
    size_t * times_inds_2 = Xinds[times_mat_index_2];

    for(size_t l=0; l<num_loops; ++l) {
        r = tidx + l * blockDim.x;

        if(x < nnz) {
            size_t const mode_i = mode_ind[x];
            size_t tmp_i = times_inds[x];
            sptScalar const entry = Xvals[x];
            size_t tmp_i_2 = times_inds_2[x];
            sptScalar tmp_val = 0;

            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            __syncthreads();
        }
    }

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops * blockDim.x;

        if(x < nnz) {
            size_t const mode_i = mode_ind[x];
            size_t tmp_i = times_inds[x];
            sptScalar const entry = Xvals[x];
            size_t tmp_i_2 = times_inds_2[x];
            sptScalar tmp_val = 0;

            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            __syncthreads();
        }
    }

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
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;
    const size_t x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    size_t const * const mode_ind = Xinds[mode];
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
      for(size_t r=0; r<R; ++r) {
        atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
      }
    }
   __syncthreads();

}



/* impl_num = 11 */
__global__ void spt_MTTKRPKernelNnz3DOneKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats) 
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }


    const size_t tidx = threadIdx.x;
    size_t x;

    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];
    size_t times_mat_index = dev_mats_order[1];
    sptScalar * times_mat = dev_mats[times_mat_index];
    size_t * times_inds = Xinds[times_mat_index];
    size_t times_mat_index_2 = dev_mats_order[2];
    sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
    size_t * times_inds_2 = Xinds[times_mat_index_2];

    for(size_t nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
            size_t const mode_i = mode_ind[x];
            size_t tmp_i = times_inds[x];
            sptScalar const entry = Xvals[x];
            size_t tmp_i_2 = times_inds_2[x];
            sptScalar tmp_val = 0;
            for(size_t r=0; r<R; ++r) {
            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
            __syncthreads();
        }
    }  

}



/* impl_num = 12 */
__global__ void spt_MTTKRPKernelRankNnz3DOneKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    size_t x;
    const size_t num_loops_r = R / blockDim.x;
    const size_t rest_loop = R - num_loops_r * blockDim.x;


    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];
    size_t times_mat_index = dev_mats_order[1];
    sptScalar * times_mat = dev_mats[times_mat_index];
    size_t * times_inds = Xinds[times_mat_index];
    size_t times_mat_index_2 = dev_mats_order[2];
    sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
    size_t * times_inds_2 = Xinds[times_mat_index_2];

    for(size_t nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
            size_t const mode_i = mode_ind[x];
            size_t tmp_i = times_inds[x];
            sptScalar const entry = Xvals[x];
            size_t tmp_i_2 = times_inds_2[x];
            sptScalar tmp_val = 0;
            size_t r;

            for(size_t l=0; l<num_loops_r; ++l) {
                r = tidy + l * blockDim.y;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidy + num_loops_r * blockDim.y;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
            __syncthreads();
        }
   
    }

}



/* impl_num = 15 */
__global__ void spt_MTTKRPKernelRankSplitNnz3DOneKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    size_t x;
    const size_t num_loops_r = R / blockDim.x;
    const size_t rest_loop = R - num_loops_r * blockDim.x;


    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];
    size_t times_mat_index = dev_mats_order[1];
    sptScalar * times_mat = dev_mats[times_mat_index];
    size_t * times_inds = Xinds[times_mat_index];
    size_t times_mat_index_2 = dev_mats_order[2];
    sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
    size_t * times_inds_2 = Xinds[times_mat_index_2];

    for(size_t nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < nnz) {
            size_t const mode_i = mode_ind[x];
            size_t tmp_i = times_inds[x];
            sptScalar const entry = Xvals[x];
            size_t tmp_i_2 = times_inds_2[x];
            sptScalar tmp_val = 0;
            size_t r;

            for(size_t l=0; l<num_loops_r; ++l) {
                r = tidx + l * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidx + num_loops_r * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
            __syncthreads();
        }
   
    }

}


/* impl_num = 16 */
__global__ void spt_MTTKRPKernelRankSplitNnzRB3DOneKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;  // index rank
    const size_t tidy = threadIdx.y;  // index nnz
    size_t x;
    const size_t num_loops_r = R / blockDim.x;
    const size_t rest_loop = R - num_loops_r * blockDim.x;
    size_t r;


    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];
    size_t times_mat_index = dev_mats_order[1];
    sptScalar * times_mat = dev_mats[times_mat_index];
    size_t * times_inds = Xinds[times_mat_index];
    size_t times_mat_index_2 = dev_mats_order[2];
    sptScalar * times_mat_2 = dev_mats[times_mat_index_2];
    size_t * times_inds_2 = Xinds[times_mat_index_2];


    for(size_t l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(size_t nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                size_t const mode_i = mode_ind[x];
                size_t tmp_i = times_inds[x];
                sptScalar const entry = Xvals[x];
                size_t tmp_i_2 = times_inds_2[x];
                sptScalar tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
        }
    }  // End for l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(size_t nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                size_t const mode_i = mode_ind[x];
                size_t tmp_i = times_inds[x];
                sptScalar const entry = Xvals[x];
                size_t tmp_i_2 = times_inds_2[x];
                sptScalar tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
        }
    }   // End if rest_loop

}


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


template <typename T>
__device__ static void print_array(const T array[], sptNnzIndex length, T start_index) {
    if(length == 0) {
        return;
    }
    printf("%d", (int) (array[0] + start_index));
    sptNnzIndex i;
    for(i = 1; i < length; ++i) {
        printf(", %d", (int) (array[i] + start_index));
    }
    printf("\n");
}


__device__ static void print_array(const sptValue array[], sptNnzIndex length, sptNnzIndex start_index) {
    if(length == 0) {
        return;
    }
    printf("%.2f", array[0] + start_index);
    sptNnzIndex i;
    for(i = 1; i < length; ++i) {
        printf(", %.2f", array[i] + start_index);
    }
    printf("\n");
}


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
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz) {
      sptIndex const mode_i = mode_ind[x];
      sptIndex times_mat_index = dev_mats_order[1];
      sptValue * times_mat = dev_mats[times_mat_index];
      sptIndex * times_inds = Xinds[times_mat_index];
      sptIndex tmp_i = times_inds[x];
      sptValue const entry = Xvals[x];
      sptIndex times_mat_index_2 = dev_mats_order[2];
      sptValue * times_mat_2 = dev_mats[times_mat_index_2];
      sptIndex * times_inds_2 = Xinds[times_mat_index_2];
      sptIndex tmp_i_2 = times_inds_2[x];
      sptValue tmp_val = 0;
      for(sptIndex r=0; r<R; ++r) {
        tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
        atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
      }

    }
   __syncthreads();

}


/* impl_num = 02 */
__global__ void spt_MTTKRPKernelNnzRank3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex tidy = threadIdx.y;
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.x + tidx;
    // printf("x: %lu, tidx: %lu, tidy: %lu\n", x, tidx, tidy);

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz && tidy < R) {
      sptIndex const mode_i = mode_ind[x];
      sptIndex times_mat_index = dev_mats_order[1];
      sptValue * times_mat = dev_mats[times_mat_index];
      sptIndex * times_inds = Xinds[times_mat_index];
      sptIndex tmp_i = times_inds[x];
      sptValue const entry = Xvals[x];
      sptIndex times_mat_index_2 = dev_mats_order[2];
      sptValue * times_mat_2 = dev_mats[times_mat_index_2];
      sptIndex * times_inds_2 = Xinds[times_mat_index_2];
      sptIndex tmp_i_2 = times_inds_2[x];
      sptValue tmp_val = 0;

      tmp_val = entry * times_mat[tmp_i * stride + tidy] * times_mat_2[tmp_i_2 * stride + tidy];
      // printf("x: %lu, tidy: %lu, tmp_val: %lf\n", x, tidy, tmp_val);
      atomicAdd(&(mvals[mode_i * stride + tidy]), tmp_val);      
    }
   __syncthreads();

}


/* impl_num = 03 */
__global__ void spt_MTTKRPKernelNnzRankSplit3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex tidy = threadIdx.y;
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.x + tidx;
    const sptIndex rank_size = R / blockDim.y;  // R is dividable to blockDim.y

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz && tidy * rank_size < R) {
      sptIndex const mode_i = mode_ind[x];
      sptIndex times_mat_index = dev_mats_order[1];
      sptValue * times_mat = dev_mats[times_mat_index];
      sptIndex * times_inds = Xinds[times_mat_index];
      sptIndex tmp_i = times_inds[x];
      sptValue const entry = Xvals[x];
      sptIndex times_mat_index_2 = dev_mats_order[2];
      sptValue * times_mat_2 = dev_mats[times_mat_index_2];
      sptIndex * times_inds_2 = Xinds[times_mat_index_2];
      sptIndex tmp_i_2 = times_inds_2[x];
      sptValue tmp_val = 0;

      for(sptIndex r=tidy*rank_size; r<(tidy+1)*rank_size; ++r) {
        tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
        atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
      }

    }
   __syncthreads();

}


/* impl_num = 04 */
__global__ void spt_MTTKRPKernelRankNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.y + tidy;

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz && tidx < R) {
      sptIndex const mode_i = mode_ind[x];
      sptIndex times_mat_index = dev_mats_order[1];
      sptValue * times_mat = dev_mats[times_mat_index];
      sptIndex * times_inds = Xinds[times_mat_index];
      sptIndex tmp_i = times_inds[x];
      sptValue const entry = Xvals[x];
      sptIndex times_mat_index_2 = dev_mats_order[2];
      sptValue * times_mat_2 = dev_mats[times_mat_index_2];
      sptIndex * times_inds_2 = Xinds[times_mat_index_2];
      sptIndex tmp_i_2 = times_inds_2[x];
      sptValue tmp_val = 0;

      tmp_val = entry * times_mat[tmp_i * stride + tidx] * times_mat_2[tmp_i_2 * stride + tidx];
      atomicAdd(&(mvals[mode_i * stride + tidx]), tmp_val);      
    }
   __syncthreads();

}


/* impl_num = 05 */
__global__ void spt_MTTKRPKernelRankSplitNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.y + tidy;
    const sptIndex num_loops = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops * blockDim.x;


    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz) {
        sptIndex const mode_i = mode_ind[x];
        sptIndex times_mat_index = dev_mats_order[1];
        sptValue * times_mat = dev_mats[times_mat_index];
        sptIndex * times_inds = Xinds[times_mat_index];
        sptIndex tmp_i = times_inds[x];
        sptValue const entry = Xvals[x];
        sptIndex times_mat_index_2 = dev_mats_order[2];
        sptValue * times_mat_2 = dev_mats[times_mat_index_2];
        sptIndex * times_inds_2 = Xinds[times_mat_index_2];
        sptIndex tmp_i_2 = times_inds_2[x];
        sptValue tmp_val = 0;
        sptIndex r;

        for(sptIndex l=0; l<num_loops; ++l) {
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
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.y + tidy;
    const sptIndex num_loops = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops * blockDim.x;
    sptIndex r;

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptIndex l=0; l<num_loops; ++l) {
        r = tidx + l * blockDim.x;

        if(x < nnz) {
            sptIndex const mode_i = mode_ind[x];
            sptIndex tmp_i = times_inds[x];
            sptValue const entry = Xvals[x];
            sptIndex tmp_i_2 = times_inds_2[x];
            sptValue tmp_val = 0;

            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            __syncthreads();
        }
    }

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops * blockDim.x;

        if(x < nnz) {
            sptIndex const mode_i = mode_ind[x];
            sptIndex tmp_i = times_inds[x];
            sptValue const entry = Xvals[x];
            sptIndex tmp_i_2 = times_inds_2[x];
            sptValue tmp_val = 0;

            tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
            atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            __syncthreads();
        }
    }

}



/* impl_num = 09 */
__global__ void spt_MTTKRPKernelScratch(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptValue * dev_scratch,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz) {
      sptIndex times_mat_index = dev_mats_order[1];
      sptValue * times_mat = dev_mats[times_mat_index];
      sptIndex * times_inds = Xinds[times_mat_index];
      sptIndex tmp_i = times_inds[x];
      sptValue const entry = Xvals[x];
      for(sptIndex r=0; r<R; ++r) {
        dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
      }

      for(sptIndex i=2; i<nmodes; ++i) {
        times_mat_index = dev_mats_order[i];
        times_mat = dev_mats[times_mat_index];
        times_inds = Xinds[times_mat_index];
        tmp_i = times_inds[x];
        for(sptIndex r=0; r<R; ++r) {
          dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
        }
      }

    }
   __syncthreads();

    if(x < nnz) {
      sptIndex const mode_i = mode_ind[x];
      // printf("x: %lu, mode_i: %lu\n", x, mode_i);
      for(sptIndex r=0; r<R; ++r) {
        atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
      }
    }
   __syncthreads();

}



/* impl_num = 11 */
__global__ void spt_MTTKRPKernelNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }


    const sptNnzIndex tidx = threadIdx.x;
    sptNnzIndex x;

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
              sptIndex const mode_i = mode_ind[x];
              sptIndex tmp_i = times_inds[x];
              sptValue const entry = Xvals[x];
              sptIndex tmp_i_2 = times_inds_2[x];
              sptValue tmp_val = 0;
              for(sptIndex r=0; r<R; ++r) {
              tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
              atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }  

}



/* impl_num = 12 */
__global__ void spt_MTTKRPKernelRankNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
            sptIndex const mode_i = mode_ind[x];
            sptIndex tmp_i = times_inds[x];
            sptValue const entry = Xvals[x];
            sptIndex tmp_i_2 = times_inds_2[x];
            sptValue tmp_val = 0;
            sptIndex r;

            for(sptIndex l=0; l<num_loops_r; ++l) {
                r = tidy + l * blockDim.y;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidy + num_loops_r * blockDim.y;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
   
    }

}



/* impl_num = 15 */
__global__ void spt_MTTKRPKernelRankSplitNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < nnz) {
            sptIndex const mode_i = mode_ind[x];
            sptIndex tmp_i = times_inds[x];
            sptValue const entry = Xvals[x];
            sptIndex tmp_i_2 = times_inds_2[x];
            sptValue tmp_val = 0;
            sptIndex r;

            for(sptIndex l=0; l<num_loops_r; ++l) {
                r = tidx + l * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidx + num_loops_r * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
   
    }

}


/* impl_num = 16 */
__global__ void spt_MTTKRPKernelRankSplitNnzRB3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptIndex r;


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];


    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                sptIndex const mode_i = mode_ind[x];
                sptIndex tmp_i = times_inds[x];
                sptValue const entry = Xvals[x];
                sptIndex tmp_i_2 = times_inds_2[x];
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }  // End for l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                sptIndex const mode_i = mode_ind[x];
                sptIndex tmp_i = times_inds[x];
                sptValue const entry = Xvals[x];
                sptIndex tmp_i_2 = times_inds_2[x];
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }   // End if rest_loop

}



/* impl_num = 21. */
__global__ void spt_MTTKRPKernelBlockNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex bidx = blockIdx.x;
    sptNnzIndex x;
    // if(tidx == 0 && bidx == 0)
    //     printf("Execute spt_MTTKRPKernelBlockNnz3D kernel.\n");

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.x) {
        num_loops_nnz = (nnz_blk + blockDim.x - 1) / blockDim.x;
    }
    // if(tidx == 0)
    //     printf("bidx: %lu, nnz_blk: %lu, nnz_blk_begin: %lu\n", bidx, nnz_blk, nnz_blk_begin);

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = tidx + nl * blockDim.x;
        if(x < nnz_blk) {
            sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_allblocks[mode];    // local base
            // printf("[x: %lu, bidx: %lu] global: %lu, mode_i: %lu\n", x, bidx, mode_ind[x + nnz_blk_begin], mode_i);
            sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
            sptValue const entry = Xvals[x + nnz_blk_begin];
            sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
            sptValue tmp_val = 0;
            // printf("[x: %lu, bidx: %lu] nnz_blk_begin: %lu, mode_ind[x + nnz_blk_begin]: %lu, mode_i: %lu, entry: %.2f, tmp_i: %lu, 1st: %.2f, tmp_i_2: %lu, 2nd: %.2f\n", x, bidx, nnz_blk_begin, mode_ind[x + nnz_blk_begin], mode_i, entry, tmp_i, times_mat[tmp_i * stride + 0], tmp_i_2, times_mat_2[tmp_i_2 * stride + 0]);
            for(sptIndex r=0; r<R; ++r) {
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }

            __syncthreads();
        }
    }   // End loop nl

}



/* impl_num = 25 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex bidx = blockIdx.x; // index block, also nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    // if(tidx == 0 && bidx == 0)
    //     printf("Execute spt_MTTKRPKernelBlockRankSplitNnz3D kernel.\n");
    

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.y) {
        num_loops_nnz = (nnz_blk + blockDim.y - 1) / blockDim.y;
    }
    // if(tidy == 0)
    //     printf("bidx: %lu, nnz_blk: %lu, nnz_blk_begin: %lu\n", bidx, nnz_blk, nnz_blk_begin);

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = tidy + nl * blockDim.y;
        if(x < nnz_blk) {
            sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_allblocks[mode];    // local base
            // printf("[x: %lu, bidx: %lu] global: %lu, mode_i: %lu\n", x, bidx, mode_ind[x + nnz_blk_begin], mode_i);
            sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
            sptValue const entry = Xvals[x + nnz_blk_begin];
            sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
            sptValue tmp_val = 0;
            sptIndex r;
            // if(tidx == 0)
            // printf("[x: %lu, bidx: %lu] nnz_blk_begin: %lu, mode_ind[tidx + nnz_blk_begin]: %lu, mode_i: %lu, entry: %.2f, tmp_i: %lu, 1st: %.2f, tmp_i_2: %lu, 2nd: %.2f\n", x, bidx, nnz_blk_begin, mode_ind[x + nnz_blk_begin], mode_i, entry, tmp_i, times_mat[tmp_i * stride + 0], tmp_i_2, times_mat_2[tmp_i_2 * stride + 0]);

            for(sptIndex l=0; l<num_loops_r; ++l) {
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




/* impl_num = 26 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnzRB3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex bidx = blockIdx.x; // index block, also nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;   
    sptIndex r; 

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.y) {
        num_loops_nnz = (nnz_blk + blockDim.y - 1) / blockDim.y;
    }

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];


    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = tidy + nl * blockDim.y;
            if(x < nnz_blk) {
                sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_allblocks[mode];    // local base
                sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
                sptValue const entry = Xvals[x + nnz_blk_begin];
                sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
                sptValue tmp_val = 0;
                
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
        }

    }   // End loop l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = tidy + nl * blockDim.y;
            if(x < nnz_blk) {
                sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_allblocks[mode];    // local base
                sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
                sptValue const entry = Xvals[x + nnz_blk_begin];
                sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r]; 
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
        }
    }   // End if rest_loop


}



/* impl_num = 35 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMCoarse(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex bidx = blockIdx.x; // index block, also nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptIndex r;
    extern __shared__ sptValue mem_pool[];
    // clock_t start_tick, end_tick;
    // double elapsed_time;
    // double g2s_time = 0, comp_time = 0, s2g_time = 0;

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    const sptIndex inds_low_mode = inds_low[bidx][mode];
    const sptIndex Xndims_blk_mode = Xndims[bidx][mode];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.y) {
        num_loops_nnz = (nnz_blk + blockDim.y - 1) / blockDim.y;
    }
    sptIndex num_loops_blk_mode = 1;
    if(Xndims_blk_mode > blockDim.y) {
        num_loops_blk_mode = (Xndims_blk_mode + blockDim.y - 1) / blockDim.y;
    }
    sptIndex sx;

    sptValue * const shr_mvals = (sptValue *) mem_pool; // size A nrows * stride 

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    /* Use registers to avoid repeated memory accesses */
    sptIndex const inds_low_allblocks_mode = inds_low_allblocks[mode];

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                shr_mvals[sx * stride + r] = 0;
            }
            __syncthreads();
        }        
    }
    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;
        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                shr_mvals[sx * stride + r] = 0;
            }
            __syncthreads();
        }
    }

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = tidy + nl * blockDim.y;
        if(x < nnz_blk) {
            sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_mode;    // local base for block
            sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
            sptValue const entry = Xvals[x + nnz_blk_begin];
            sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
            sptValue tmp_val = 0;
            // printf("[x: %lu, bidx: %lu] entry: %f, 1st: %f, 2nd: %f\n", x, bidx, entry, times_mat[tmp_i * stride + 0], times_mat_2[tmp_i_2 * stride + 0]);

            for(sptIndex l=0; l<num_loops_r; ++l) {
                r = tidx + l * blockDim.x;

                // start_tick = clock();
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(shr_mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
                // elapsed_time = (clock() - start_tick)/0.82e9;
                // comp_time += elapsed_time;

            } // End loop l

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidx + num_loops_r * blockDim.x;

                // start_tick = clock();
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r]; 
                atomicAdd(&(shr_mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
                // elapsed_time = (clock() - start_tick)/0.82e9;
                // comp_time += elapsed_time;
            } // End rest_loop

        }   // End if(x < nnz_blk)
    }   // End loop nl

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                atomicAdd(&(mvals[(sx + inds_low_mode - inds_low_allblocks_mode) * stride + r]), shr_mvals[sx * stride + r]);
            }
            __syncthreads();
        }
    }
    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;
        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                atomicAdd(&(mvals[(sx + inds_low_mode - inds_low_allblocks_mode) * stride + r]), shr_mvals[sx * stride + r]);
            }
            __syncthreads();
        }
    }
    // printf("(%u, <%u, %u>)  g2s_time: %lf, comp_time: %lf, s2g_time: %lf\n", bidx, tidx, tidy, g2s_time, comp_time, s2g_time);
}




/* impl_num = 36 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMCoarseRB(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex bidx = blockIdx.x; // index block, also nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptIndex r;
    extern __shared__ sptValue mem_pool[];
    // clock_t start_tick, end_tick;
    // double elapsed_time;
    // double g2s_time = 0, comp_time = 0, s2g_time = 0;

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    const sptIndex inds_low_mode = inds_low[bidx][mode];
    const sptIndex Xndims_blk_mode = Xndims[bidx][mode];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.y) {
        num_loops_nnz = (nnz_blk + blockDim.y - 1) / blockDim.y;
    }
    sptIndex num_loops_blk_mode = 1;
    if(Xndims_blk_mode > blockDim.y) {
        num_loops_blk_mode = (Xndims_blk_mode + blockDim.y - 1) / blockDim.y;
    }
    sptIndex sx;

    sptValue * const shr_mvals = (sptValue *) mem_pool; // size A nrows * stride 

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    /* Use registers to avoid repeated memory accesses */
    sptIndex const inds_low_allblocks_mode = inds_low_allblocks[mode];

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                shr_mvals[sx * stride + r] = 0;
            }
            __syncthreads();
        }        

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = tidy + nl * blockDim.y;
            if(x < nnz_blk) {
                sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_mode;    // local base for block
                sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
                sptValue const entry = Xvals[x + nnz_blk_begin];
                sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(shr_mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();
            }
        }

        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                atomicAdd(&(mvals[(sx + inds_low_mode - inds_low_allblocks_mode) * stride + r]), shr_mvals[sx * stride + r]);
            }
            __syncthreads();
        }

    }   // End loop l: num_loops_r


    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                shr_mvals[sx * stride + r] = 0;
            }
            __syncthreads();
        }

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = tidy + nl * blockDim.y;
            if(x < nnz_blk) {
                sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_mode;    // local base for block
                sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index];  // local base
                sptValue const entry = Xvals[x + nnz_blk_begin];
                sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_allblocks[times_mat_index_2];  // local base
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r]; 
                atomicAdd(&(shr_mvals[mode_i * stride + r]), tmp_val);
                __syncthreads();

            }
        }

        for(sptIndex sl=0; sl<num_loops_blk_mode; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_mode) {
                atomicAdd(&(mvals[(sx + inds_low_mode - inds_low_allblocks_mode) * stride + r]), shr_mvals[sx * stride + r]);
            }
            __syncthreads();
        }

    }   // End if rest_loop

}




/* impl_num = 45 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMMedium(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex bidx = blockIdx.x; // index block, also nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptIndex r;
    extern __shared__ sptValue mem_pool[];

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    sptIndex * const inds_low_blk = inds_low[bidx];
    sptIndex * const Xndims_blk = Xndims[bidx];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.y) {
        num_loops_nnz = (nnz_blk + blockDim.y - 1) / blockDim.y;
    }   


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    /* Use registers to avoid repeated memory accesses */
    sptIndex const inds_low_blk_A = inds_low_blk[mode];
    sptIndex const inds_low_blk_B = inds_low_blk[times_mat_index];
    sptIndex const inds_low_blk_C = inds_low_blk[times_mat_index_2];
    sptIndex const inds_low_allblocks_A = inds_low_allblocks[mode];
    sptIndex const inds_low_allblocks_B = inds_low_allblocks[times_mat_index];
    sptIndex const inds_low_allblocks_C = inds_low_allblocks[times_mat_index_2];
    sptIndex const Xndims_blk_A = Xndims_blk[mode];
    sptIndex const Xndims_blk_B = Xndims_blk[times_mat_index];
    sptIndex const Xndims_blk_C = Xndims_blk[times_mat_index_2];

    sptIndex num_loops_blk_A = 1;
    if(Xndims_blk_A > blockDim.y) {
        num_loops_blk_A = (Xndims_blk_A + blockDim.y - 1) / blockDim.y;
    }
    sptIndex num_loops_blk_B = 1;
    if(Xndims_blk_B > blockDim.y) {
        num_loops_blk_B = (Xndims_blk_B + blockDim.y - 1) / blockDim.y;
    }
    sptIndex num_loops_blk_C = 1;
    if(Xndims_blk_C > blockDim.y) {
        num_loops_blk_C = (Xndims_blk_C + blockDim.y - 1) / blockDim.y;
    }   
    sptIndex sx;


    // if(tidx == 0 && tidy == 0)
    //     printf("[%lu, (%lu, %lu)]  (Xndims_blk_A: %lu, Xndims_blk_B: %lu, Xndims_blk_C: %lu); (inds_low_blk_A: %lu, inds_low_blk_B: %lu, inds_low_blk_C: %lu); (inds_low_allblocks_A: %lu, inds_low_allblocks_B: %lu, inds_low_allblocks_C: %lu)\n", bidx, tidx, tidy, Xndims_blk_A, Xndims_blk_B, Xndims_blk_C, inds_low_blk_A, inds_low_blk_B, inds_low_blk_C, inds_low_allblocks_A, inds_low_allblocks_B, inds_low_allblocks_C);


    sptValue * const shrA = (sptValue *) mem_pool; // A: size nrows * stride
    sptValue * const shrB = (sptValue *) (shrA + Xndims_blk_A * stride); // B: size nrows * stride
    sptValue * const shrC = (sptValue *) (shrB + Xndims_blk_B * stride); // C: size nrows * stride

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        /* Set shrA = 0 */
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                shrA[sx * stride + r] = 0;
            }
            __syncthreads();
        }
        /* Load shrB */
        for(sptIndex sl=0; sl<num_loops_blk_B; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_B) {
                shrB[sx * stride + r] = times_mat[(sx + inds_low_blk_B - inds_low_allblocks_B) * stride + r];
            }
            __syncthreads();
        }
        /* Load shrC */
        for(sptIndex sl=0; sl<num_loops_blk_C; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_C) {
                shrC[sx * stride + r] = times_mat_2[(sx + inds_low_blk_C - inds_low_allblocks_C) * stride + r];
            }
            __syncthreads();
        }

    }   // End loop l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;
        /* Set shrA = 0 */
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                shrA[sx * stride + r] = 0;
            }
            __syncthreads();
        }
        /* Load shrB */
        for(sptIndex sl=0; sl<num_loops_blk_B; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_B) {
                shrB[sx * stride + r] = times_mat[(sx + inds_low_blk_B - inds_low_allblocks_B) * stride + r];
            }
            __syncthreads();
        }
        /* Load shrC */
        for(sptIndex sl=0; sl<num_loops_blk_C; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_C) {
                shrC[sx * stride + r] = times_mat_2[(sx + inds_low_blk_C - inds_low_allblocks_C) * stride + r];
            }
            __syncthreads();
        }
    }
    // __syncthreads();


    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = tidy + nl * blockDim.y;
        if(x < nnz_blk) {
            sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_blk_A;    // local base for block
            // printf("[x: %lu, bidx: %lu] global: %lu, mode_i: %lu\n", x, bidx, mode_ind[x + nnz_blk_begin], mode_i);
            sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_blk_B;  // local base
            sptValue const entry = Xvals[x + nnz_blk_begin];
            sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_blk_C;  // local base
            sptValue tmp_val = 0;

            for(sptIndex l=0; l<num_loops_r; ++l) {
                r = tidx + l * blockDim.x;            
                // if(tidx == 0)
                //     printf("[%lu, (0, %lu)]  nnz_blk_begin: %lu, mode_ind[tidy + nnz_blk_begin]: %lu, mode_i: %lu, entry: %.2f, tmp_i: %lu, 1st: %.2f, tmp_i_2: %lu, 2nd: %.2f\n", bidx, tidy, nnz_blk_begin, mode_ind[tidy + nnz_blk_begin], mode_i, entry, tmp_i, shrB[tmp_i * stride + 0], tmp_i_2, shrC[tmp_i_2 * stride + 0]);

                tmp_val = entry * shrB[tmp_i * stride + r] * shrC[tmp_i_2 * stride + r];
                atomicAdd(&(shrA[mode_i * stride + r]), tmp_val);
                __syncthreads();
            } // End loop l: num_loops_r

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidx + num_loops_r * blockDim.x;

                tmp_val = entry * shrB[tmp_i * stride + r] * shrC[tmp_i_2 * stride + r];
                atomicAdd(&(shrA[mode_i * stride + r]), tmp_val);
                __syncthreads();
            } // End if rest_loop
        }   // End if(x < nnz_blk)

    }   // End loop nl: num_loops_nnz

    /* Store back shrA */
    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                atomicAdd(&(mvals[(sx + inds_low_blk_A - inds_low_allblocks_A) * stride + r]), shrA[sx * stride + r]);
            }
            __syncthreads();
        }
    }
    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                atomicAdd(&(mvals[(sx + inds_low_blk_A - inds_low_allblocks_A) * stride + r]), shrA[sx * stride + r]);
            }
            __syncthreads();
        }
    }


}




/* impl_num = 46 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMMediumRB(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats) 
{
    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    const sptNnzIndex bidx = blockIdx.x; // index block, also nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptIndex r;
    extern __shared__ sptValue mem_pool[];

    /* block range */
    const sptNnzIndex nnz_blk = nnz[bidx];
    const sptNnzIndex nnz_blk_begin = dev_nnz_blk_begin[bidx];
    sptIndex * const inds_low_blk = inds_low[bidx];
    sptIndex * const Xndims_blk = Xndims[bidx];
    sptNnzIndex num_loops_nnz = 1;
    if(nnz_blk > blockDim.y) {
        num_loops_nnz = (nnz_blk + blockDim.y - 1) / blockDim.y;
    }   


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    /* Use registers to avoid repeated memory accesses */
    sptIndex const inds_low_blk_A = inds_low_blk[mode];
    sptIndex const inds_low_blk_B = inds_low_blk[times_mat_index];
    sptIndex const inds_low_blk_C = inds_low_blk[times_mat_index_2];
    sptIndex const inds_low_allblocks_A = inds_low_allblocks[mode];
    sptIndex const inds_low_allblocks_B = inds_low_allblocks[times_mat_index];
    sptIndex const inds_low_allblocks_C = inds_low_allblocks[times_mat_index_2];
    sptIndex const Xndims_blk_A = Xndims_blk[mode];
    sptIndex const Xndims_blk_B = Xndims_blk[times_mat_index];
    sptIndex const Xndims_blk_C = Xndims_blk[times_mat_index_2];

    sptIndex num_loops_blk_A = 1;
    if(Xndims_blk_A > blockDim.y) {
        num_loops_blk_A = (Xndims_blk_A + blockDim.y - 1) / blockDim.y;
    }
    sptIndex num_loops_blk_B = 1;
    if(Xndims_blk_B > blockDim.y) {
        num_loops_blk_B = (Xndims_blk_B + blockDim.y - 1) / blockDim.y;
    }
    sptIndex num_loops_blk_C = 1;
    if(Xndims_blk_C > blockDim.y) {
        num_loops_blk_C = (Xndims_blk_C + blockDim.y - 1) / blockDim.y;
    }   
    sptIndex sx;


    // if(tidx == 0 && tidy == 0)
    //     printf("[%lu, (%lu, %lu)]  (Xndims_blk_A: %lu, Xndims_blk_B: %lu, Xndims_blk_C: %lu); (inds_low_blk_A: %lu, inds_low_blk_B: %lu, inds_low_blk_C: %lu); (inds_low_allblocks_A: %lu, inds_low_allblocks_B: %lu, inds_low_allblocks_C: %lu)\n", bidx, tidx, tidy, Xndims_blk_A, Xndims_blk_B, Xndims_blk_C, inds_low_blk_A, inds_low_blk_B, inds_low_blk_C, inds_low_allblocks_A, inds_low_allblocks_B, inds_low_allblocks_C);


    sptValue * const shrA = (sptValue *) mem_pool; // A: size nrows * stride
    sptValue * const shrB = (sptValue *) (shrA + Xndims_blk_A * stride); // B: size nrows * stride
    sptValue * const shrC = (sptValue *) (shrB + Xndims_blk_B * stride); // C: size nrows * stride

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        /* Set shrA = 0 */
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                shrA[sx * stride + tidx] = 0;
            }
            __syncthreads();
        }
        /* Load shrB */
        for(sptIndex sl=0; sl<num_loops_blk_B; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_B) {
                shrB[sx * stride + tidx] = times_mat[(sx + inds_low_blk_B - inds_low_allblocks_B) * stride + r];
            }
            __syncthreads();
        }
        /* Load shrC */
        for(sptIndex sl=0; sl<num_loops_blk_C; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_C) {
                shrC[sx * stride + tidx] = times_mat_2[(sx + inds_low_blk_C - inds_low_allblocks_C) * stride + r];
            }
            __syncthreads();
        }


        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = tidy + nl * blockDim.y;
            if(x < nnz_blk) {
                sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_blk_A;    // local base for block
                sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_blk_B;  // local base
                sptValue const entry = Xvals[x + nnz_blk_begin];
                sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_blk_C;  // local base
                sptValue tmp_val = 0;

                tmp_val = entry * shrB[tmp_i * stride + tidx] * shrC[tmp_i_2 * stride + tidx];
                atomicAdd(&(shrA[mode_i * stride + tidx]), tmp_val);
                __syncthreads();

            }
        }

        /* Store back shrA */
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                atomicAdd(&(mvals[(sx + inds_low_blk_A - inds_low_allblocks_A) * stride + r]), shrA[sx * stride + tidx]);
            }
            __syncthreads();
        }

    }   // End loop l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        /* Set shrA = 0 */
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                shrA[sx * stride + tidx] = 0;
            }
            __syncthreads();
        }
        /* Load shrB */
        for(sptIndex sl=0; sl<num_loops_blk_B; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_B) {
                shrB[sx * stride + tidx] = times_mat[(sx + inds_low_blk_B - inds_low_allblocks_B) * stride + r];
            }
            __syncthreads();
        }
        /* Load shrC */
        for(sptIndex sl=0; sl<num_loops_blk_C; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_C) {
                shrC[sx * stride + tidx] = times_mat_2[(sx + inds_low_blk_C - inds_low_allblocks_C) * stride + r];
            }
            __syncthreads();
        }


        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = tidy + nl * blockDim.y;
            if(x < nnz_blk) {
                sptIndex const mode_i = mode_ind[x + nnz_blk_begin] - inds_low_blk_A;    // local base for block
                sptIndex tmp_i = times_inds[x + nnz_blk_begin] - inds_low_blk_B;  // local base
                sptValue const entry = Xvals[x + nnz_blk_begin];
                sptIndex tmp_i_2 = times_inds_2[x + nnz_blk_begin] - inds_low_blk_C;  // local base
                sptValue tmp_val = 0;

                tmp_val = entry * shrB[tmp_i * stride + tidx] * shrC[tmp_i_2 * stride + tidx];
                atomicAdd(&(shrA[mode_i * stride + tidx]), tmp_val);
                __syncthreads();

            }
        }


        /* Store back shrA */
        for(sptIndex sl=0; sl<num_loops_blk_A; ++sl) {
            sx = tidy + sl * blockDim.y;
            if(sx < Xndims_blk_A) {
                atomicAdd(&(mvals[(sx + inds_low_blk_A - inds_low_allblocks_A) * stride + r]), shrA[sx * stride + tidx]);
            }
            __syncthreads();
        }
    }   // End if rest_loop

}








/* impl_num = 59 */
__global__ void spt_MTTKRPKernelScratchDist(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    const sptIndex * inds_low,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptValue * dev_scratch
) {
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex x = blockIdx.x * blockDim.x + tidx;

    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = dev_mats[nmodes];

    // if(x == 0) {
    //     printf("mvals:\n");
    //     for(sptIndex i=0; i<Xndims[mode]; ++i) {
    //         printf("%lf\n", mvals[i * stride]);
    //     }
    //     printf("mvals end\n");
    // }

    if(x < nnz) {
        sptIndex times_mat_index = dev_mats_order[1];
        sptValue * times_mat = dev_mats[times_mat_index];
        sptIndex * times_inds = Xinds[times_mat_index];
        sptIndex tmp_i = times_inds[x] - inds_low[times_mat_index];
        sptValue const entry = Xvals[x];
        for(sptIndex r=0; r<R; ++r) {
            dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
        }

        for(sptIndex i=2; i<nmodes; ++i) {
            times_mat_index = dev_mats_order[i];
            times_mat = dev_mats[times_mat_index];
            times_inds = Xinds[times_mat_index];
            tmp_i = times_inds[x] - inds_low[times_mat_index];
            for(sptIndex r=0; r<R; ++r) {
                dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
            }
        }

    }

    __syncthreads();

    if(x < nnz) {
        sptIndex const mode_i = mode_ind[x] - inds_low[mode];
        // printf("x: %lu, mode_ind[x]: %lu, mode_i: %lu\n", x, mode_ind[x], mode_i);
        for(sptIndex r=0; r<R; ++r) {
            atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
        }
    }
    __syncthreads();

    // if(x == 0) {
    //     printf("inds_low[mode]: %lu, Xndims[mode]: %lu\n", inds_low[mode], Xndims[mode]);
    //     printf("nnz: %lu\n", nnz);;
    //     printf("mvals:\n");
    //     for(sptIndex i=0; i<Xndims[mode]; ++i) {
    //         printf("%lf\n", mvals[i * stride]);
    //     }
    //     printf("mvals end\n");
    // }
    
}






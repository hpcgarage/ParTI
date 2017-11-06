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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "sptensor.h"


/* impl_num = 01 */
__global__ void spt_TTMNaiveKernel(
    sptValue *Y_val, sptIndex Y_stride, sptNnzIndex Y_nnz,
    const sptValue *X_val, sptNnzIndex X_nnz, const sptIndex *X_inds_m,
    const sptNnzIndex *fiberidx_val, sptNnzIndex fiberidx_len,
    const sptValue *U_val, sptIndex U_nrows, sptIndex U_ncols, sptIndex U_stride,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex tidy = threadIdx.y;
    const sptNnzIndex i = (blockIdx.x + block_offset) * blockDim.x + tidx;

    if(i >= Y_nnz || tidy >= U_ncols) return;
    const sptNnzIndex inz_begin = fiberidx_val[i];
    const sptNnzIndex inz_end = fiberidx_val[i+1];

    Y_val[i*Y_stride + tidy] = 0;
    for(sptNnzIndex j = inz_begin; j < inz_end; ++j) {
        const sptIndex r = X_inds_m[j];
        Y_val[i*Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
    }
}


/* impl_num = 02 */
__global__ void spt_TTMKernel(
    sptValue *Y_val, sptIndex Y_stride, sptNnzIndex Y_nnz,
    const sptValue *X_val, sptNnzIndex X_nnz, const sptIndex *X_inds_m,
    const sptNnzIndex *fiberidx_val, sptNnzIndex fiberidx_len,
    const sptValue *U_val, sptIndex U_nrows, sptIndex U_ncols, sptIndex U_stride,
    sptNnzIndex block_offset) 
{
    extern __shared__ sptValue mem_pool[];

    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex tidy = threadIdx.y;
    const sptNnzIndex i = (blockIdx.x + block_offset) * blockDim.x + tidx;
    //const sptNnzIndex off = blockIdx.x * blockDim.x + tidx;

    sptNnzIndex inz_begin, inz_end;
    if(i < Y_nnz) {
        inz_begin = fiberidx_val[i];
        inz_end = fiberidx_val[i+1];
    }
    __syncthreads();

    //sptValue * const Y_shr = (sptValue *) &mem_pool[tidx*Y_stride]; // size U_ncols
    sptValue * const Y_shr = (sptValue *) mem_pool; // size U_ncols
    if(i < Y_nnz && tidy < U_ncols) {
        Y_shr[tidx * Y_stride + tidy] = 0;
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        for(sptNnzIndex j = inz_begin; j < inz_end; ++j) {
            const sptIndex r = X_inds_m[j];
            Y_shr[tidx * Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
        }
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        Y_val[i*Y_stride + tidy] = Y_shr[tidx*Y_stride + tidy];
    }
    __syncthreads();
}


/* impl_num = 11 */
__global__ void spt_TTMNnzKernel(
    sptValue *Y_val, 
    sptIndex Y_stride, 
    sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;
    sptNnzIndex x;

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const sptNnzIndex inz_begin = fiberidx_val[x];
            const sptNnzIndex inz_end = fiberidx_val[x+1];

            for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                const sptIndex row = X_inds_m[i];
                for(sptIndex r=0; r<U_ncols; ++r) {
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                }
            }
        }
        __syncthreads();
    }

}


/* impl_num = 12 */
__global__ void spt_TTMNnzRankKernel(
    sptValue *Y_val, 
    sptIndex Y_stride, 
    sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;    // Index nnz
    const sptNnzIndex tidy = threadIdx.y;    // Index rank
    sptNnzIndex x;
    const sptIndex num_loops_r = U_ncols / blockDim.y;
    const sptIndex rest_loop = U_ncols - num_loops_r * blockDim.y;

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const sptNnzIndex inz_begin = fiberidx_val[x];
            const sptNnzIndex inz_end = fiberidx_val[x+1];
            sptIndex r;

            for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                const sptIndex row = X_inds_m[i];
                for(sptIndex l=0; l<num_loops_r; ++l) {
                    r = tidy + l * blockDim.y;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }

                if(rest_loop > 0 && tidy < rest_loop) {
                    r = tidy + num_loops_r * blockDim.y;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
        __syncthreads();
    }

}


/* impl_num = 13 */
__global__ void spt_TTMRankNnzKernel(
    sptValue *Y_val, 
    sptIndex Y_stride, 
    sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;    // Index rank
    const sptNnzIndex tidy = threadIdx.y;    // Index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = U_ncols / blockDim.x;
    const sptIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    sptIndex r;

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const sptNnzIndex inz_begin = fiberidx_val[x];
            const sptNnzIndex inz_end = fiberidx_val[x+1];

            for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                const sptIndex row = X_inds_m[i];
                for(sptIndex l=0; l<num_loops_r; ++l) {
                    r = tidx + l * blockDim.x;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }

                if(rest_loop > 0 && tidx < rest_loop) {
                    r = tidx + num_loops_r * blockDim.x;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
        __syncthreads();
    }
}


/* impl_num = 14 */
__global__ void spt_TTMRankRBNnzKernel(
    sptValue *Y_val, 
    sptIndex Y_stride, 
    sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;    // Index rank
    const sptNnzIndex tidy = threadIdx.y;    // Index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = U_ncols / blockDim.x;
    const sptIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    sptIndex r;

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];

                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];

                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

}


/* impl_num = 15 */
__global__ void spt_TTMRankRBNnzKernelSM(
    sptValue *Y_val, 
    sptIndex Y_stride, sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride) 
{
    extern __shared__ sptValue mem_pool[];
    sptValue * const Y_shr = (sptValue *) mem_pool; // size U_ncols

    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }
    
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex tidy = threadIdx.y;
    sptNnzIndex x;
    const sptIndex num_loops_r = U_ncols / blockDim.x;
    const sptIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    sptIndex r;


    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];
                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }


    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];
                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }

}
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
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, const size_t *X_inds_m,
    const size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset) 
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;

    if(i >= Y_nnz || tidy >= U_ncols) return;
    const size_t inz_begin = fiberidx_val[i];
    const size_t inz_end = fiberidx_val[i+1];

    Y_val[i*Y_stride + tidy] = 0;
    for(size_t j = inz_begin; j < inz_end; ++j) {
        const size_t r = X_inds_m[j];
        Y_val[i*Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
    }
}


/* impl_num = 02 */
__global__ void spt_TTMKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, const size_t *X_inds_m,
    const size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset) 
{
    extern __shared__ sptScalar mem_pool[];

    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;
    //const size_t off = blockIdx.x * blockDim.x + tidx;

    size_t inz_begin, inz_end;
    if(i < Y_nnz) {
        inz_begin = fiberidx_val[i];
        inz_end = fiberidx_val[i+1];
    }
    __syncthreads();

    //sptScalar * const Y_shr = (sptScalar *) &mem_pool[tidx*Y_stride]; // size U_ncols
    sptScalar * const Y_shr = (sptScalar *) mem_pool; // size U_ncols
    if(i < Y_nnz && tidy < U_ncols) {
        Y_shr[tidx * Y_stride + tidy] = 0;
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        for(size_t j = inz_begin; j < inz_end; ++j) {
            const size_t r = X_inds_m[j];
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
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;
    size_t x;

    for(size_t nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const size_t inz_begin = fiberidx_val[x];
            const size_t inz_end = fiberidx_val[x+1];

            for(size_t i = inz_begin; i < inz_end; ++i) {
                const size_t row = X_inds_m[i];
                for(size_t r=0; r<U_ncols; ++r) {
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                }
            }
        }
        __syncthreads();
    }

}


/* impl_num = 12 */
__global__ void spt_TTMNnzRankKernel(
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;    // Index nnz
    const size_t tidy = threadIdx.y;    // Index rank
    size_t x;
    const size_t num_loops_r = U_ncols / blockDim.y;
    const size_t rest_loop = U_ncols - num_loops_r * blockDim.y;

    for(size_t nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const size_t inz_begin = fiberidx_val[x];
            const size_t inz_end = fiberidx_val[x+1];
            size_t r;

            for(size_t i = inz_begin; i < inz_end; ++i) {
                const size_t row = X_inds_m[i];
                for(size_t l=0; l<num_loops_r; ++l) {
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
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;    // Index rank
    const size_t tidy = threadIdx.y;    // Index nnz
    size_t x;
    const size_t num_loops_r = U_ncols / blockDim.x;
    const size_t rest_loop = U_ncols - num_loops_r * blockDim.x;
    size_t r;

    for(size_t nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const size_t inz_begin = fiberidx_val[x];
            const size_t inz_end = fiberidx_val[x+1];

            for(size_t i = inz_begin; i < inz_end; ++i) {
                const size_t row = X_inds_m[i];
                for(size_t l=0; l<num_loops_r; ++l) {
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
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride)
{
    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const size_t tidx = threadIdx.x;    // Index rank
    const size_t tidy = threadIdx.y;    // Index nnz
    size_t x;
    const size_t num_loops_r = U_ncols / blockDim.x;
    const size_t rest_loop = U_ncols - num_loops_r * blockDim.x;
    size_t r;

    for(size_t l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(size_t nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const size_t inz_begin = fiberidx_val[x];
                const size_t inz_end = fiberidx_val[x+1];

                for(size_t i = inz_begin; i < inz_end; ++i) {
                    const size_t row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(size_t nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const size_t inz_begin = fiberidx_val[x];
                const size_t inz_end = fiberidx_val[x+1];

                for(size_t i = inz_begin; i < inz_end; ++i) {
                    const size_t row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

}


/* impl_num = 15 */
__global__ void spt_TTMRankRBNnzKernelSM(
    sptScalar *Y_val, 
    size_t Y_stride, size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride) 
{
    extern __shared__ sptScalar mem_pool[];
    sptScalar * const Y_shr = (sptScalar *) mem_pool; // size U_ncols

    size_t num_loops_nnz = 1;
    size_t const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }
    
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    size_t x;
    const size_t num_loops_r = U_ncols / blockDim.x;
    const size_t rest_loop = U_ncols - num_loops_r * blockDim.x;
    size_t r;


    for(size_t l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(size_t nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const size_t inz_begin = fiberidx_val[x];
                const size_t inz_end = fiberidx_val[x+1];
                for(size_t i = inz_begin; i < inz_end; ++i) {
                    const size_t row = X_inds_m[i];
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

        for(size_t nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const size_t inz_begin = fiberidx_val[x];
                const size_t inz_end = fiberidx_val[x+1];
                for(size_t i = inz_begin; i < inz_end; ++i) {
                    const size_t row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }

}
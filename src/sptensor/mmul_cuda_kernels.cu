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
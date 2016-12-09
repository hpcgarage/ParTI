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
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../error/error.h"
#include <cuda_runtime.h>


__global__ static void spt_MatrixDotMulSeqKernel(
    size_t const mode,
    size_t const nmodes, 
    const size_t rank, 
    const size_t stride, 
    sptScalar ** dev_ata)
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;

    sptScalar * ovals = dev_ata[nmodes];
    ovals[tidx * stride + tidy] = 1;
    __syncthreads();

    for(size_t m=1; m < nmodes; ++m) {
        size_t const pm = (mode + m) % nmodes;
        sptScalar const * vals = dev_ata[pm];
        ovals[tidx * stride + tidy] *= vals[tidx * stride + tidy];
    }
    __syncthreads();
}


int sptCudaMatrixDotMulSeq(
    size_t const mode,
    size_t const nmodes, 
    const size_t rank, 
    const size_t stride, 
    sptScalar ** dev_ata)
{
    dim3 nthreads(rank, rank);  // rank <=  16
    dim3 nblocks(1, 1);

    spt_MatrixDotMulSeqKernel<<<nblocks, nthreads>>> (mode, nmodes, rank, stride, dev_ata);
    
    int result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA Matrix sptCudaMatrixDotMulSeq");

    return 0;
}



__global__ static void spt_Matrix2NormKernel(
    size_t const nrows,
    size_t const ncols,
    size_t const stride,
    sptScalar * const dev_vals,
    sptScalar * const dev_lambda)
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t bidx = blockIdx.x;
    const size_t i = bidx * blockDim.x + tidx;

    if(i < nrows)
        atomicAdd(&(dev_lambda[tidy]), dev_vals[i*stride + tidy] * dev_vals[i*stride + tidy]);
    __syncthreads();

    dev_lambda[tidy] = sqrt(dev_lambda[tidy]);
    __syncthreads();

    if(i < nrows)
        dev_vals[i*stride + tidy] /= dev_lambda[tidy];
    __syncthreads();

}



int sptCudaMatrix2Norm(
    size_t const nrows,
    size_t const ncols,
    size_t const stride,
    sptScalar * const dev_vals,
    sptScalar * const dev_lambda)
{
    dim3 nthreads(16, ncols);  // ncols <=  16
    dim3 nblocks((nrows + 16 -1) / 16);

    spt_Matrix2NormKernel<<<nblocks, nthreads>>>(nrows, ncols, stride, dev_vals, dev_lambda);
    int result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA Matrix sptCudaMatrix2Norm");

    return 0;
}

__global__ static void spt_IdentityMatrixKernel(
    size_t const nrows,
    size_t const ncols,
    size_t const stride,
    sptScalar * const dev_vals)
{
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;

    dev_vals[tidx * stride + tidy] = 0;
    __syncthreads();
    if (tidx == tidy)
        dev_vals[tidx * stride + tidy] = 1;
    __syncthreads();
}



int sptCudaIdentityMatrix(size_t const nrows, size_t const ncols, size_t const stride, sptScalar * const dev_vals)
{
    assert(nrows <= 16 && ncols <= 16);
    spt_IdentityMatrixKernel<<<nrows, ncols>>>(nrows, ncols, stride, dev_vals);
    int result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA Matrix sptCudaIdentityMatrix");

    return 0;
}

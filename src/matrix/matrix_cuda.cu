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
    sptIndex const mode,
    sptIndex const nmodes, 
    sptIndex const rank, 
    sptIndex const stride, 
    sptValue ** dev_ata)
{
    const sptIndex tidx = (sptIndex)threadIdx.x;
    const sptIndex tidy = (sptIndex)threadIdx.y;

    sptValue * ovals = dev_ata[nmodes];
    ovals[tidx * stride + tidy] = 1;
    __syncthreads();

    for(sptIndex m=1; m < nmodes; ++m) {
        sptIndex const pm = (mode + m) % nmodes;
        sptValue const * vals = dev_ata[pm];
        ovals[tidx * stride + tidy] *= vals[tidx * stride + tidy];
    }
    __syncthreads();
}


int sptCudaMatrixDotMulSeq(
    sptIndex const mode,
    sptIndex const nmodes, 
    sptIndex const rank, 
    sptIndex const stride, 
    sptValue ** dev_ata)
{
    dim3 nthreads(rank, rank);  // rank <=  16
    dim3 nblocks(1, 1);

    spt_MatrixDotMulSeqKernel<<<nblocks, nthreads>>> (mode, nmodes, rank, stride, dev_ata);
    
    int result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA Matrix sptCudaMatrixDotMulSeq");

    return 0;
}



__global__ static void spt_Matrix2NormKernel(
    sptIndex const nrows,
    sptIndex const ncols,
    sptIndex const stride,
    sptValue * const dev_vals,
    sptValue * const dev_lambda)
{
    const sptIndex tidx = (sptIndex)threadIdx.x;
    const sptIndex tidy = (sptIndex)threadIdx.y;
    const sptIndex bidx = (sptIndex)blockIdx.x;
    const sptIndex i = bidx * blockDim.x + tidx;

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
    sptIndex const nrows,
    sptIndex const ncols,
    sptIndex const stride,
    sptValue * const dev_vals,
    sptValue * const dev_lambda)
{
    dim3 nthreads(16, ncols);  // ncols <=  16
    dim3 nblocks((nrows + 16 -1) / 16);

    spt_Matrix2NormKernel<<<nblocks, nthreads>>>(nrows, ncols, stride, dev_vals, dev_lambda);
    int result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA Matrix sptCudaMatrix2Norm");

    return 0;
}


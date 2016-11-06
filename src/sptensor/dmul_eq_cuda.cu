/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <SpTOL.h>
#include "sptensor.h"

__global__ static void spt_DotMulKernel(size_t nnz, sptScalar *Z_val, sptScalar *X_val, sptScalar *Y_val) 
{
    const size_t tidx = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + tidx;

    if(i < nnz) {
        Z_val[i] = X_val[i] * Y_val[i];
    }
    __syncthreads();
}



/**
 * CUDA parallelized Element wise multiply two sparse tensors, with exactly the same nonzero
 * distribution.
 * @param[out] Z the result of X*Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptCudaSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {
    size_t i;
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns DotMul", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns DotMul", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(Y->nnz != X->nnz) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "nonzero distribution mismatch");
    }
    size_t nnz = X->nnz;

    sptCopySparseTensor(Z, X);

    sptScalar *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    sptScalar *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * sizeof (sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaMemcpy(Y_val, Y->values.data, Y->nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    sptScalar *Z_val = NULL;
    result = cudaMalloc((void **) &Z_val, X->nnz * sizeof (sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaMemset(Z_val, 0, X->nnz * sizeof (sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    size_t nthreads = 128;
    size_t nblocks = (nnz + nthreads -1)/ nthreads;

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    spt_DotMulKernel<<<nblocks, nthreads>>>(nnz, Z_val, X_val, Y_val);
    result = cudaThreadSynchronize();

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CUDA  SpTns DotMul");
    sptFreeTimer(timer);

    cudaMemcpy(Z->values.data, Z_val, Z->nnz * sizeof (sptScalar), cudaMemcpyDeviceToHost);

    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaFree(Z_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z);
    return 0;
}

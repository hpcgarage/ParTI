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

__global__ static void spt_TTMKernel(
    sptValue *Y_val,
    const sptValue *X_val,
    sptIndex XY_stride,
    sptNnzIndex XY_nnz,
    const sptValue *U_val,
    sptIndex U_nrows, sptIndex U_ncols, sptIndex U_stride,
    sptIndex mode
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < XY_nnz) {
        size_t r, k;
        for(k = 0; k < U_ncols; ++k) {
            Y_val[tid*XY_stride + k] = 0;
            for(r = 0; r < U_nrows; ++r) {
                Y_val[tid*XY_stride + k] += X_val[tid*XY_stride + r] * U_val[r*U_stride + k];
            }
        }
    }
}

static sptNnzIndex spt_GetBlockCount(sptNnzIndex threads) {
    return (threads / 256) + ((threads & 255) != 0);
}

int sptCudaSemiSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    const sptSemiSparseTensor *X,
    const sptMatrix *U,
    sptIndex mode
) {
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    ind_buf = new sptIndex[X->nmodes * sizeof *ind_buf];
    if(!ind_buf) {
        return -1;
    }
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    delete[] ind_buf;
    if(result) {
        return result;
    }
    for(m = 0; m < Y->nmodes; ++m) {
        if(m != mode) {
            sptFreeIndexVector(&Y->inds[m]);
            result = sptCopyIndexVector(&Y->inds[m], &X->inds[m]);
            if(result != 0) {
                return result;
            }
        }
    }
    result = sptResizeMatrix(&Y->values, X->nnz);
    if(result != 0) {
        return result;
    }
    Y->nnz = X->nnz;

    sptNnzIndex blocks_count = spt_GetBlockCount(Y->nnz);
    sptNnzIndex threads_count = blocks_count * 256;
    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, threads_count * Y->stride * sizeof (sptValue));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, threads_count * X->stride * sizeof (sptValue));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    cudaMemcpy(X_val, X->values.values, X->nnz * X->stride * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (sptValue));
    if(result != 0) {
        return result;
    }
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (sptValue), cudaMemcpyHostToDevice);

    spt_TTMKernel<<<blocks_count, 256>>>(Y_val, X_val, Y->stride, Y->nnz, U_val, U->nrows, U->ncols, U->stride, mode);
    result = cudaGetLastError();
    if(result != 0) {
        return result;
    }

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (sptValue), cudaMemcpyDeviceToHost);
    cudaFree(U_val); cudaFree(X_val); cudaFree(Y_val);

    return 0;
}

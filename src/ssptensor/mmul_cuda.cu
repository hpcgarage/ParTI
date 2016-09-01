#include <SpTOL.h>

__global__ static void spt_TTMKernel(
    sptScalar *Y_val,
    const sptScalar *X_val,
    size_t XY_stride,
    size_t XY_nnz,
    const sptScalar *U_val,
    size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t mode
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

static size_t spt_GetBlockCount(size_t threads) {
    return (threads / 256) + ((threads & 255) != 0);
}

int sptCudaSemiSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    const sptSemiSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
    int result;
    size_t *ind_buf;
    size_t m;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    ind_buf = new size_t[X->nmodes * sizeof *ind_buf];
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
            sptFreeSizeVector(&Y->inds[m]);
            result = sptCopySizeVector(&Y->inds[m], &X->inds[m]);
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

    size_t blocks_count = spt_GetBlockCount(Y->nnz);
    size_t threads_count = blocks_count * 256;
    sptScalar *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, threads_count * Y->stride * sizeof (sptScalar));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    sptScalar *X_val = NULL;
    result = cudaMalloc((void **) &X_val, threads_count * X->stride * sizeof (sptScalar));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    cudaMemcpy(X_val, X->values.values, X->nnz * X->stride * sizeof (sptScalar), cudaMemcpyHostToDevice);
    sptScalar *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (sptScalar));
    if(result != 0) {
        return result;
    }
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (sptScalar), cudaMemcpyHostToDevice);

    spt_TTMKernel<<<blocks_count, 256>>>(Y_val, X_val, Y->stride, Y->nnz, U_val, U->nrows, U->ncols, U->stride, mode);
    result = cudaGetLastError();
    if(result != 0) {
        return result;
    }

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    cudaFree(U_val); cudaFree(X_val); cudaFree(Y_val);

    return 0;
}

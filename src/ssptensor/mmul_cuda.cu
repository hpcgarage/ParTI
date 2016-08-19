#include <SpTOL.h>

__global__ static void spt_TTMKernel(
    sptScalar *Y_val,
    const sptScalar *X_val,
    size_t XY_stride,
    const sptScalar *U_val,
    size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t mode
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r, k;
    for(k = 0; k < U_nrows; ++k) {
        Y_val[tid*XY_stride + k] = 0;
        for(r = 0; r < U_ncols; ++r) {
            Y_val[tid*XY_stride + k] += X_val[tid*XY_stride + r] * U_val[r*U_stride + k];
        }
    }
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

    sptScalar *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * Y->stride * sizeof (sptScalar));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    sptScalar *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * X->stride * sizeof (sptScalar));
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

    spt_TTMKernel<<<Y->nnz, 1>>>(Y_val, X_val, Y->stride, U_val, U->nrows, U->ncols, U->stride, mode);

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    cudaFree(U_val); cudaFree(X_val); cudaFree(X_val);

    return 0;
}

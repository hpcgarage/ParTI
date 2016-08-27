#include <SpTOL.h>

__global__ static void spt_TTMKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_ncols, size_t U_stride
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < Y_nnz) {
        size_t inz_begin = fiberidx_val[tid];
        size_t inz_end = fiberidx_val[tid+1];
        size_t j, k;
        for(k = 0; k < U_ncols; ++k) {
            Y_val[tid*Y_stride + k] = 0;
            for(j = inz_begin; j < inz_end; ++j) {
                size_t r = X_inds_m[j];
                Y_val[tid*Y_stride + k] += X_val[j] * U_val[r*U_stride + k];
            }
        }
    }
}

static size_t spt_GetBlockCount(size_t threads) {
    return (threads / 256) + ((threads & 255) != 0);
}

int sptCudaSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    sptSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
    int result;
    size_t *ind_buf;
    size_t m;
    sptSizeVector fiberidx;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    if(X->sortkey != mode) {
        sptSparseTensorSortIndexAtMode(X, mode);
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
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);

    size_t blocks_count = spt_GetBlockCount(Y->nnz);
    sptScalar *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * Y->stride * sizeof (sptScalar));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    sptScalar *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptScalar));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    size_t *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, X->nnz * sizeof (size_t));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    cudaMemcpy(X_inds_m, X->inds[mode].data, X->nnz * sizeof (size_t), cudaMemcpyHostToDevice);
    sptScalar *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (sptScalar));
    if(result != 0) {
        return result;
    }
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (sptScalar), cudaMemcpyHostToDevice);
    size_t *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (size_t));
    if(result != 0) {
        return result;
    }
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (size_t), cudaMemcpyHostToDevice);

    spt_TTMKernel<<<blocks_count, 256>>>(
        Y_val, Y->stride, Y->nnz,
        X_val, X->nnz, X_inds_m,
        fiberidx_val, fiberidx.len,
        U_val, U->ncols, U->stride
    );

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    cudaFree(fiberidx_val); cudaFree(U_val); cudaFree(X_inds_m); cudaFree(X_val); cudaFree(Y_val);
    sptFreeSizeVector(&fiberidx);

    return 0;
}

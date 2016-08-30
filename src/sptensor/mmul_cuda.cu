#include <SpTOL.h>

__global__ static void spt_TTMKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride
) {
    extern __shared__ char mem_pool[];
    const size_t bid = blockIdx.x;
    const size_t tid = threadIdx.x;
    sptScalar *const Y_shr = (sptScalar *) &mem_pool[0]; // size U_ncols
    sptScalar *const X_shr = (sptScalar *) &mem_pool[U_ncols * sizeof (sptScalar)]; // size U_nrows
    size_t *const r_shr = (size_t *) &mem_pool[(U_ncols+U_nrows) * sizeof (sptScalar)]; // size U_nrows

    const size_t inz_begin = fiberidx_val[bid];
    const size_t inz_end = fiberidx_val[bid+1];
    size_t i;
    // Fill Y_shr with 0, length U_ncols
    Y_shr[tid] = 0;
    // Fill X_shr with X_val, length inz_end-inz_begin
    for(i = tid; i < inz_end-inz_begin; i += blockDim.x) {
        X_shr[i] = X_val[i+inz_begin];
    }
    // Fill r_shr with X_inds_m, length U_ncols
    for(i = tid; i < inz_end-inz_begin; i += blockDim.x) {
        r_shr[i] = X_inds_m[i+inz_begin];
    }
    __syncthreads();
    // Do calculations, U_ncols threads
    for(i = 0; i < inz_end-inz_begin; ++i) {
        Y_shr[tid] += X_shr[i] * U_val[r_shr[i]*U_stride + tid];
    }
    __syncthreads();
    // Write back from Y_shr, length U_ncols
    for(i = tid; i < U_ncols; ++i) {
        Y_val[bid*Y_stride + i] = Y_shr[i];
    }
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

    size_t sharedMem = (Y->ndims[mode] + X->ndims[mode])*sizeof (sptScalar) + X->ndims[mode]*sizeof (size_t);

    spt_TTMKernel<<<Y->nnz, U->ncols, sharedMem>>>(
        Y_val, Y->stride, Y->nnz,
        X_val, X->nnz, X_inds_m,
        fiberidx_val, fiberidx.len,
        U_val, U->nrows, U->ncols, U->stride
    );

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    cudaFree(fiberidx_val); cudaFree(U_val); cudaFree(X_inds_m); cudaFree(X_val); cudaFree(Y_val);
    sptFreeSizeVector(&fiberidx);

    return 0;
}

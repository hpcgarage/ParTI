__global__ void spt_TTMNaiveKernel(
    double *Y_val, size_t Y_stride, size_t Y_nnz,
    const double *X_val, size_t X_nnz, const size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const double *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset
) {
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

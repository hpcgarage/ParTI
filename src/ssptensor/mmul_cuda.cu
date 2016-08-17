#include <SpTOL.h>


int sptCudaSemiSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    const sptSemiSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
    int result;
    size_t *ind_buf;
    size_t m;
    if(mode != X->mode) {
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
    if(result) {
        delete[] ind_buf;
        return result;
    }
    return -1;
}

__global__ static void spt_TTMKernel(
    sptScalar *Y_val,
    const sptScalar *X_val,
    size_t X_ncols,
    const sptScalar *U_val,
    size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t mode
) {
    size_t r, k;
    for(r = 0; r < X_ncols; ++r) {
        Y_val[r] = 0;
        for(k = 0; k < U_nrows; ++k) {
            Y_val[k] += X_val[r] * U_val[r*U_stride + k];
        }
    }
}

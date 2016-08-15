#include <SpTOL.h>


int sptCudaSparseTensorMulMatrixAsSemiSparseTensor(
    sptSemiSparseTensor *Y,
    const sptSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
    cudaSetDevice(0);
    int result;
    size_t *ind_buf;
    size_t m, i;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    if(!ind_buf) {
        return -1;
    }
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    if(result) {
        free(ind_buf);
        return result;
    }
    // TODO
    return -1;
}


__global__ static void spt_TTMKernel(
    sptSemiSparseTensor *Y,
    const sptSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {

}

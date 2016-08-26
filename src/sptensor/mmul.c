#include <SpTOL.h>
#include <stdlib.h>

int sptSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, size_t mode) {
    int result;
    size_t *ind_buf;
    size_t m, i;
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
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    if(!ind_buf) {
        return -1;
    }
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    // jli: use pre-processing to allocate Y size outside this function.
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    if(result) {
        return result;
    }
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);
    for(i = 0; i < Y->nnz; ++i) {
        size_t k;
        for(k = 0; k < U->ncols; ++k) {
            size_t inz_begin = fiberidx.data[i];
            size_t inz_end = fiberidx.data[i+1];
            size_t j;
            for(j = inz_begin; j < inz_end; ++j) {
                size_t r = Y->inds[mode].data[j];
                Y->values.values[i*Y->stride + k] += X->values.data[j] * U->values[r*U->stride + k];
            }
        }
    }
    return 0;
}

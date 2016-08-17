#include <SpTOL.h>
#include "ssptensor.h"
#include <stdlib.h>

int sptSemiSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    const sptSemiSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
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
    free(ind_buf);
    if(result) {
        return result;
    }
    for(i = 0; i < X->nnz; ++i) {
        size_t r, k;
        for(m = 0; m < X->nmodes; ++m) {
            if(m != mode) {
                result = sptAppendSizeVector(&Y->inds[m], X->inds[m].data[i]);
                if(result) {
                    free(ind_buf);
                    return result;
                }
            }
        }
        result = sptAppendMatrix(&Y->values, NULL);
        if(result) {
            return result;
        }
        memset(&Y->values.values[i*Y->stride], 0, Y->stride * sizeof (sptScalar));
        for(r = 0; r < U->ncols; ++r) {
            for(k = 0; k < U->nrows; ++k) {
                Y->values.values[i*Y->stride + k] += X->values.values[i*X->stride + r] * U->values[r*U->stride + k];
            }
        }
        ++Y->nnz;
    }
    return 0;
}

#include <SpTOL.h>
#include <stdlib.h>

static int spt_SparseTensorFindIndex(size_t *index, const sptSparseTensor *tsr, const size_t *inds) {
    size_t l = 0;
    size_t r = tsr->nnz;
    while(l < r) {
        size_t p = (l+r) / 2;
        size_t m;
        for(m = 0; m < tsr->nmodes; ++m) {
            size_t eleind1 = tsr->inds[m].data[p];
            size_t eleind2 = inds[m];
            if(eleind1 < eleind2) {
                l = p+1;
                break;
            } else if(eleind1 > eleind2) {
                r = p;
                break;
            } else {
                *index = p;
                return 1;
            }
        }
    }
    *index = l;
    return 0;
}

/* jli: (TODO) Change to a sparse tensor times a dense matrix.
    Output a "semi-sparse" tensor in the timing mode.
    This function can be kept for the future. */
int sptSparseTensorMulMatrix(sptSparseTensor *Y, const sptSparseTensor *X, const sptMatrix *U, size_t mode) {
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
    result = sptNewSparseTensor(Y, X->nmodes, ind_buf);
    if(result) {
        free(ind_buf);
        return result;
    }
    for(i = 0; i < X->nnz; ++i) {
        size_t r;
        for(m = 0; m < X->nmodes; ++m) {
            ind_buf[m] = X->inds[m].data[i];
        }
        for(r = 0; r < U->ncols; ++r) {
            // (sb) TODO: Some O(n) algorithm?
            sptScalar acc_val = X->values.data[i] * U->values[X->inds[mode].data[i]*U->stride + r];
            size_t acc_idx;
            ind_buf[mode] = r;
            if(spt_SparseTensorFindIndex(&acc_idx, Y, ind_buf)) {
                Y->values.data[acc_idx] += acc_val;
            } else {
                for(m = 0; m < X->nmodes; ++m) {
                    result = sptAppendSizeVector(&Y->inds[m], ind_buf[m]);
                    if(result) {
                        free(ind_buf);
                        return result;
                    }
                }
                result = sptAppendVector(&Y->values, acc_val);
                if(result) {
                    free(ind_buf);
                    return result;
                }
                sptSparseTensorSortIndex(Y);
            }
        }
    }
    free(ind_buf);
    return 0;
}

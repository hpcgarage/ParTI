#include <SpTOL.h>
#include <stdlib.h>

/* jli: (TODO) Change to a sparse tensor times a dense matrix. 
    Output a "semi-sparse" tensor in the timing mode.
    This function can be kept for the future. */
int sptSparseTensorMulMatrix(sptSparseTensor *Y, const sptSparseTensor *X, const sptMatrix *U, size_t mode) {
    int result;
    sptSparseTensor XT;
    size_t *ndims;
    size_t m;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    XT.nmodes = X->nmodes;
    XT.ndims = malloc(XT.nmodes * sizeof *XT.ndims);
    if(!XT.ndims) {
        return -1;
    }
    for(m = 0; m < XT.nmodes; ++m) {
        if(m < mode) {
            XT.ndims[m] = X->ndims[m];
        } else if(m == mode) {
            XT.ndims[XT.nmodes-1] = X->ndims[m];
        } else {
            XT.ndims[m-1] = X->ndims[m];
        }
    }
    XT.nnz = X->nnz;
    XT.inds = malloc(XT.nmodes * sizeof *XT.inds);
    if(!XT.inds) {
        free(XT.ndims);
        return -1;
    }
    for(m = 0; m < XT.nmodes; ++m) {
        if(m < mode) {
            result = sptCopySizeVector(&XT.inds[m], &X->inds[m]);
        } else if(m == mode) {
            result = sptCopySizeVector(&XT.inds[XT.nmodes-1], &X->inds[m]);
        } else {
            result = sptCopySizeVector(&XT.inds[m-1], &X->inds[m]);
        }
        if(result) {
            free(XT.inds);
            free(XT.ndims);
            return result;
        }
    }
    result = sptCopyVector(&XT.values, &X->values);
    sptSparseTensorSortIndex(&XT);
    ndims = malloc(XT.nmodes * sizeof *ndims);
    if(!ndims) {
        sptFreeSparseTensor(&XT);
        return result;
    }
    for(m = 0; m < XT.nmodes; ++m) {
        if(m < mode) {
            ndims[m] = X->ndims[m];
        } else if(m == mode) {
            ndims[XT.nmodes-1] = U->ncols;
        } else {
            ndims[m-1] = X->ndims[m];
        }
    }
    result = sptNewSparseTensor(Y, XT.nmodes, ndims);
    free(ndims);
    if(result) {
        sptFreeSparseTensor(&XT);
        return result;
    }
    // TODO
    sptFreeSparseTensor(&XT);
    return 0;
}

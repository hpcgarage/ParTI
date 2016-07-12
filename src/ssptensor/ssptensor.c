#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, size_t nmodes, const size_t ndims[]) {
    size_t i;
    int result;
    if(nmodes < 2) {
        return -1;
    }
    tsr->nmodes = nmodes;
    tsr->ndims = malloc((nmodes-1) * sizeof *tsr->ndims);
    if(!tsr->ndims) {
        return -1;
    }
    memcpy(tsr->ndims, ndims, (nmodes-1) * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc((nmodes-1) * sizeof *tsr->inds);
    if(!tsr->inds) {
        return -1;
    }
    for(i = 0; i < nmodes-1; ++i) {
        result = sptNewSizeVector(&tsr->inds[i], 0, 0);
        if(result) {
            return result;
        }
    }
    result = sptNewVector(&tsr->values, 0, 0);
    if(result) {
        return result;
    }
    return 0;
}

int sptCopySemiSparseTensor(sptSemiSparseTensor *dest, const sptSemiSparseTensor *src) {
    size_t i;
    int result;
    assert(src->nmodes >= 2);
    dest->nmodes = src->nmodes;
    dest->ndims = malloc((dest->nmodes-1) * sizeof *dest->ndims);
    if(!dest->ndims) {
        return -1;
    }
    memcpy(dest->ndims, src->ndims, (src->nmodes-1) * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc((dest->nmodes-1) * sizeof *dest->inds);
    if(!dest->inds) {
        return -1;
    }
    for(i = 0; i < dest->nmodes-1; ++i) {
        result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        if(result) {
            return result;
        }
    }
    result = sptCopyVector(&dest->values, &src->values);
    if(result) {
        return result;
    }
    return 0;
}

void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr) {
    size_t i;
    for(i = 0; i < tsr->nmodes-1; ++i) {
        sptFreeSizeVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeVector(&tsr->values);
}

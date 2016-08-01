#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, size_t nmodes, size_t mode, const size_t ndims[]) {
    size_t i;
    int result;
    if(nmodes < 2) {
        return -1;
    }
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    if(!tsr->ndims) {
        return -1;
    }
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->mode = mode;
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    if(!tsr->inds) {
        return -1;
    }
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&tsr->inds[i], 0, 0);
        if(result) {
            return result;
        }
    }
    tsr->stride = ((ndims[mode]-1)/8+1)*8;
    result = sptNewMatrix(&tsr->values, 0, tsr->stride);
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
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    if(!dest->ndims) {
        return -1;
    }
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->mode = src->mode;
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    if(!dest->inds) {
        return -1;
    }
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        if(result) {
            return result;
        }
    }
    dest->stride = src->stride;
    result = sptCopyMatrix(&dest->values, &src->values);
    if(result) {
        return result;
    }
    return 0;
}

void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeSizeVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeMatrix(&tsr->values);
}

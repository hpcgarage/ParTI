#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptNewSparseTensor(sptSparseTensor *tsr, size_t nmodes, const size_t ndims[]) {
    size_t i;
    int result;
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    if(!tsr->ndims) {
        return -1;
    }
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
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
    result = sptNewVector(&tsr->values, 0, 0);
    if(result) {
        return result;
    }
    return 0;
}

int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    dest->nmodes = src->nmodes;
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    if(!dest->ndims) {
        return -1;
    }
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
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
    result = sptCopyVector(&dest->values, &src->values);
    if(result) {
        return result;
    }
    return 0;
}

void sptFreeSparseTensor(sptSparseTensor *tsr) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeSizeVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeVector(&tsr->values);
}

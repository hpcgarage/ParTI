#include <SpTOL.h>
#include "ssptensor.h"
#include <stdlib.h>
#include <string.h>

int sptSparseTensorToSemiSparseTensor(sptSemiSparseTensor *dest, const sptSparseTensor *src, size_t mode) {
    size_t i;
    int result;
    size_t nmodes = src->nmodes;
    if(nmodes < 2) {
        return -1;
    }
    dest->nmodes = nmodes;
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    if(!dest->ndims) {
        return -1;
    }
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->mode = mode;
    dest->nnz = src->nnz;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    if(!dest->inds) {
        return -1;
    }
    for(i = 0; i < nmodes; ++i) {
        if(i == mode) {
            result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        } else {
            result = sptNewSizeVector(&dest->inds[i], 0, 0);
        }
        if(result) {
            return result;
        }
    }
    dest->stride = ((dest->ndims[mode]-1)/8+1)*8;
    result = sptNewMatrix(&dest->values, dest->nnz, dest->stride);
    if(result) {
        return result;
    }
    for(i = 0; i < dest->nnz; ++i) {
        dest->values.values[i*dest->stride + src->inds[mode].data[i]] = src->values.data[i];
    }
    sptSemiSparseTensorSortIndex(dest);
    spt_SemiSparseTensorMergeValues(dest);
    return 0;
}

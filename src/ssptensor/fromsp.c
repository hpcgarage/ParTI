#include <SpTOL.h>
#include "ssptensor.h"
#include <stdlib.h>
#include <string.h>

int sptSparseTensorToSemiSparseTensor(sptSemiSparseTensor *dest, const sptSparseTensor *src, size_t mode) {
    size_t i;
    int result;
    size_t nmodes = src->nmodes;
    if(nmodes < 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns -> SspTns", "nmodes < 2");
    }
    dest->nmodes = nmodes;
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns -> SspTns");
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->mode = mode;
    dest->nnz = src->nnz;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SpTns -> SspTns");
    for(i = 0; i < nmodes; ++i) {
        if(i != mode) {
            result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        } else {
            result = sptNewSizeVector(&dest->inds[i], 0, 0);
        }
        spt_CheckError(result, "SpTns -> SspTns", NULL);
    }
    result = sptNewMatrix(&dest->values, dest->nnz, dest->ndims[mode]);
    spt_CheckError(result, "SpTns -> SspTns", NULL);
    dest->stride = dest->values.stride;
    memset(dest->values.values, 0, dest->nnz * dest->stride * sizeof (sptScalar));
    for(i = 0; i < dest->nnz; ++i) {
        dest->values.values[i*dest->stride + src->inds[mode].data[i]] = src->values.data[i];
    }
    sptSemiSparseTensorSortIndex(dest);
    spt_SemiSparseTensorMergeValues(dest);
    return 0;
}

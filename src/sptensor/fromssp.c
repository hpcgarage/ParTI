#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptSemiSparseTensorToSparseTensor(sptSparseTensor *dest, const sptSemiSparseTensor *src) {
    size_t i;
    int result;
    size_t nmodes = src->nmodes;
    dest->nmodes = nmodes;
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    if(!dest->ndims) {
        return -1;
    }
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->nnz = 0;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    if(!dest->inds) {
        return -1;
    }
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&dest->inds[i], 0, src->nnz);
        if(result) {
            return result;
        }
    }
    result = sptNewVector(&dest->values, 0, src->nnz);
    if(result) {
        return result;
    }
    for(i = 0; i < src->nnz; ++i) {
        size_t j;
        for(j = 0; j < src->ndims[nmodes-1]; ++j) {
            sptScalar data = src->values.data[i*src->stride + j];
            if(data != 0) {
                size_t m;
                for(m = 0; m < nmodes-1; ++m) {
                    result = sptAppendSizeVector(&dest->inds[m], src->inds[m].data[i]);
                    if(result) {
                        return result;
                    }
                }
                result = sptAppendSizeVector(&dest->inds[nmodes-1], j);
                if(result) {
                    return result;
                }
                result = sptAppendVector(&dest->values, data);
                if(result) {
                    return result;
                }
            }
        }
    }
    return 0;
}

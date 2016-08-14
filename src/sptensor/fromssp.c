#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptSemiSparseTensorToSparseTensor(sptSparseTensor *dest, const sptSemiSparseTensor *src, sptScalar epsilon) {
    size_t i;
    int result;
    size_t nmodes = src->nmodes;
    assert(epsilon > 0);
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
        for(j = 0; j < src->ndims[src->mode]; ++j) {
            sptScalar data = src->values.values[i*src->stride + j];
            if(!(data < epsilon && data > -epsilon)) {
                size_t m;
                for(m = 0; m < nmodes; ++m) {
                    if(m != src->mode) {
                        result = sptAppendSizeVector(&dest->inds[m], src->inds[m].data[i]);
                    } else {
                        result = sptAppendSizeVector(&dest->inds[src->mode], j);
                    }
                    if(result) {
                        return result;
                    }
                }
                result = sptAppendVector(&dest->values, data);
                if(result) {
                    return result;
                }
                ++dest->nnz;
            }
        }
    }
    sptSparseTensorSortIndex(dest);
    return 0;
}

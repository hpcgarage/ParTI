#include <assert.h>
#include <SpTOL.h>
#include "ssptensor.h"
#include "../sptensor/sptensor.h"

int sptSemiSparseTensorSetIndices(
    sptSemiSparseTensor *dest,
    sptSizeVector *fiberidx,
    sptSparseTensor *ref
) {
    size_t lastidx = ref->nnz;
    size_t i, m;
    int result;
    assert(dest->nmodes == ref->nmodes);
    if(ref->sortkey != dest->mode) {
        sptSparseTensorSortIndexAtMode(ref, dest->mode);
    }
    result = sptNewSizeVector(fiberidx, 0, 0);
    if(result != 0) {
        return result;
    }
    for(i = 0; i < ref->nnz; ++i) {
        if(lastidx == ref->nnz || spt_SparseTensorCompareIndices(ref, lastidx, ref, i) != 0) {
            for(m = 0; m < dest->nmodes; ++m) {
                if(m != dest->mode) {
                    result = sptAppendSizeVector(&dest->inds[m], ref->inds[m].data[i]);
                    if(result != 0) {
                        return result;
                    }
                }
            }
            lastidx = i;
            ++dest->nnz;
            if(fiberidx != NULL) {
                result = sptAppendSizeVector(fiberidx, i);
                if(result != 0) {
                    return result;
                }
            }
        }
    }
    if(fiberidx != NULL) {
        result = sptAppendSizeVector(fiberidx, ref->nnz);
        if(result != 0) {
            return result;
        }
    }
    result = sptResizeMatrix(&dest->values, dest->nnz);
    if(result != 0) {
        return result;
    }
    memset(dest->values.values, 0, dest->nnz * dest->stride * sizeof (sptScalar));
    return 0;
}

#include <assert.h>
#include <SpTOL.h>
#include "ssptensor.h"
#include "../sptensor/sptensor.h"

static int spt_SparseTensorCompareExceptMode(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2, size_t mode);

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
    spt_CheckError(result, "SspTns SetIndices", NULL);
    dest->nnz = 0;
    for(i = 0; i < ref->nnz; ++i) {
        if(lastidx == ref->nnz || spt_SparseTensorCompareExceptMode(ref, lastidx, ref, i, dest->mode) != 0) {
            for(m = 0; m < dest->nmodes; ++m) {
                if(m != dest->mode) {
                    result = sptAppendSizeVector(&dest->inds[m], ref->inds[m].data[i]);
                    spt_CheckError(result, "SspTns SetIndices", NULL);
                }
            }
            lastidx = i;
            ++dest->nnz;
            if(fiberidx != NULL) {
                result = sptAppendSizeVector(fiberidx, i);
                spt_CheckError(result, "SspTns SetIndices", NULL);
            }
        }
    }
    if(fiberidx != NULL) {
        result = sptAppendSizeVector(fiberidx, ref->nnz);
        spt_CheckError(result, "SspTns SetIndices", NULL);
    }
    result = sptResizeMatrix(&dest->values, dest->nnz);
    spt_CheckError(result, "SspTns SetIndices", NULL);
    memset(dest->values.values, 0, dest->nnz * dest->stride * sizeof (sptScalar));
    return 0;
}

static int spt_SparseTensorCompareExceptMode(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2, size_t mode) {
    size_t i;
    size_t eleind1, eleind2;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != mode) {
            eleind1 = tsr1->inds[i].data[ind1];
            eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    return 0;
}

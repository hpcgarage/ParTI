#include <SpTOL.h>
#include "ssptensor.h"
#include <stdlib.h>
#include <string.h>

static void spt_QuickSortIndex(sptSemiSparseTensor *tsr, size_t l, size_t r, sptScalar buffer[]);
static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]);

int sptSemiSparseTensorSortIndex(sptSemiSparseTensor *tsr) {
    sptScalar *buffer = malloc(tsr->stride * sizeof (sptScalar));
    spt_CheckOSError(!buffer, "SspTns SortIndex");
    spt_QuickSortIndex(tsr, 0, tsr->nnz, buffer);
    free(buffer);
    return 0;
}

static void spt_QuickSortIndex(sptSemiSparseTensor *tsr, size_t l, size_t r, sptScalar buffer[]) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SemiSparseTensorCompareIndices(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SemiSparseTensorCompareIndices(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j, buffer);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    spt_QuickSortIndex(tsr, l, i, buffer);
    spt_QuickSortIndex(tsr, i, r, buffer);
}

static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            size_t eleind1 = tsr->inds[i].data[ind1];
            size_t eleind2 = tsr->inds[i].data[ind2];
            tsr->inds[i].data[ind1] = eleind2;
            tsr->inds[i].data[ind2] = eleind1;
        }
    }
    if(ind1 != ind2) {
        memcpy(buffer, &tsr->values.values[ind1*tsr->stride], tsr->stride * sizeof (sptScalar));
        memcpy(&tsr->values.values[ind1*tsr->stride], &tsr->values.values[ind2*tsr->stride], tsr->stride * sizeof (sptScalar));
        memcpy(&tsr->values.values[ind2*tsr->stride], buffer, tsr->stride * sizeof (sptScalar));
    }
}

#include <SpTOL.h>
#include "sptensor.h"

static void spt_QuickSortIndex(sptSparseTensor *tsr, size_t l, size_t r);
static void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2);

void sptSparseTensorSortIndex(sptSparseTensor *tsr) {
    spt_QuickSortIndex(tsr, 0, tsr->nnz);
}

static void spt_QuickSortIndex(sptSparseTensor *tsr, size_t l, size_t r) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndices(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndices(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    spt_QuickSortIndex(tsr, l, i);
    spt_QuickSortIndex(tsr, i, r);
}

static void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2) {
    size_t i;
    sptScalar val1, val2;
    for(i = 0; i < tsr->nmodes; ++i) {
        size_t eleind1 = tsr->inds[i].data[ind1];
        size_t eleind2 = tsr->inds[i].data[ind2];
        tsr->inds[i].data[ind1] = eleind2;
        tsr->inds[i].data[ind2] = eleind1;
    }
    val1 = tsr->values.data[ind1];
    val2 = tsr->values.data[ind2];
    tsr->values.data[ind2] = val1;
    tsr->values.data[ind1] = val2;
}

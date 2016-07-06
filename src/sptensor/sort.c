#include <SpTOL.h>
#include "sptensor.h"

static void spt_QuickSortIndex(sptSparseTensor *tsr, size_t l, size_t r);
static int spt_CompareIndices(const sptSparseTensor *tsr, size_t ind1, size_t ind2);
static void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2);

void spt_SparseTensorSortIndex(sptSparseTensor *tsr) {
    spt_QuickSortIndex(tsr, 0, tsr->nnz);
}

static void spt_QuickSortIndex(sptSparseTensor *tsr, size_t l, size_t r) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_CompareIndices(tsr, i, p) < 0) {
            ++i;
        }
        while(spt_CompareIndices(tsr, p, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
    }
    spt_QuickSortIndex(tsr, l, i);
    spt_QuickSortIndex(tsr, i, r);
}

static int spt_CompareIndices(const sptSparseTensor *tsr, size_t ind1, size_t ind2) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        size_t eleind1 = tsr->inds[i].data[ind1];
        size_t eleind2 = tsr->inds[i].data[ind2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
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

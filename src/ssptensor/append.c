#include <SpTOL.h>
#include "ssptensor.h"

static int spt_SemiSparseTensorCompareIndices(const sptSemiSparseTensor *tsr, size_t el_idx, const size_t indices[]) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            if(tsr->indices[i].data[el_idx] < indices) {
                return -1;
            } else if(tsr->indices[i].data[el_idx] > indices) {
                return 1;
            }
        }
    }
    return 0;
}

int spt_SemiSparseTensorAppend(sptSemiSparseTensor *tsr, const size_t indices[], sptScalar value) {
    int need_resize = 0;
    if(tsr->nnz == 0) {
        need_resize = 1;
    } else if(spt_SemiSparseTensorCompareIndices(tsr, tsr->nnz-1, indices) != 0) {
        need_resize = 1;
    }
    if(need_resize) {
        for(i = 0; i < tsr->nmodes; ++i) {
            if(i != tsr->mode) {
                sptAppendSizeVector(&tsr->indices[i], indices[i]);
            }
        }
        sptAppendMatrix(&tsr->values, NULL);
        memset(tsr->values.values[tsr->nnz * tsr->stride], 0, tsr->ncols * sizeof (spScalar));
        ++tsr->nnz;
    }
    tsr->values.values[(tsr->nnz-1) * tsr->stride + indices[tsr->mode]] = value;
}

#include <SpTOL.h>
#include "ssptensor.h"
#include <string.h>

static int spt_CompareIndices(const sptSemiSparseTensor *tsr, size_t el_idx, const size_t indices[]) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            if(tsr->inds[i].data[el_idx] < indices[i]) {
                return -1;
            } else if(tsr->inds[i].data[el_idx] > indices[i]) {
                return 1;
            }
        }
    }
    return 0;
}

int spt_SemiSparseTensorAppend(sptSemiSparseTensor *tsr, const size_t indices[], sptScalar value) {
    int result;
    int need_resize = 0;
    if(tsr->nnz == 0) {
        need_resize = 1;
    } else if(spt_CompareIndices(tsr, tsr->nnz-1, indices) != 0) {
        need_resize = 1;
    }
    if(need_resize) {
        size_t i;
        for(i = 0; i < tsr->nmodes; ++i) {
            if(i != tsr->mode) {
                result = sptAppendSizeVector(&tsr->inds[i], indices[i]);
                if(result) {
                    return result;
                }
            }
        }
        result = sptAppendMatrix(&tsr->values, NULL);
        if(result) {
            return result;
        }
        memset(&tsr->values.values[tsr->nnz * tsr->stride], 0, tsr->nmodes * sizeof (sptScalar));
        ++tsr->nnz;
    }
    tsr->values.values[(tsr->nnz-1) * tsr->stride + indices[tsr->mode]] = value;
    return 0;
}

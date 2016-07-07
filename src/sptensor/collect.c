#include <SpTOL.h>
#include "sptensor.h"

/**
 * Collect zeros from a sparse tensor
 *
 * This function finds zero values from the sparse tensor,
 * move them to the end of the array, then truncate the array.
 * Orders are not kept, call sptSparseTensorSortIndex after this.
 */
void spt_SparseTensorCollectZeros(sptSparseTensor *tsr) {
    size_t i =  0;
    size_t nnz = tsr->nnz;
    size_t mode;
    while(i < nnz) {
        if(tsr->values.data[i] == 0) {
            for(mode = 0; mode < tsr->nmodes; ++mode) {
                tsr->inds[mode].data[i] = tsr->inds[mode].data[nnz-1];
            }
            tsr->values.data[i] = tsr->values.data[nnz-1];
            --nnz;
        } else {
            ++i;
        }
    }
    tsr->nnz = nnz;
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->inds[mode].len = nnz;
    }
    tsr->values.len = nnz;
}

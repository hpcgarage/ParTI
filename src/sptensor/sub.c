#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorSub(sptSparseTensor *Y, const sptSparseTensor *X) {
    size_t i, j, Ynnz;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        return -1;
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            return -1;
        }
    }
    /* Add elements one by one, assume indices are ordered */
    i = 0;
    j = 0;
    Ynnz = Y->nnz;
    while(i < X->nnz && j < Ynnz) {
        int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
        if(compare > 0) {
            ++j;
        } else if(compare < 0) {
            size_t mode;
            int result;
            for(mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Y->inds[mode], X->inds[mode].data[i]);
                if(result) {
                    return result;
                }
            }
            result = sptAppendVector(&Y->values, -X->values.data[i]);
            if(result) {
                return result;
            }
            ++Y->nnz;
            ++i;
        } else {
            Y->values.data[j] -= X->values.data[i];
            ++i;
            ++j;
        }
    }
    /* Append remaining elements of X to Y */
    while(i < X->nnz) {
        size_t mode;
        int result;
        for(mode = 0; mode < X->nmodes; ++mode) {
            result = sptAppendSizeVector(&Y->inds[mode], X->inds[mode].data[i]);
            if(result) {
                return result;
            }
        }
        result = sptAppendVector(&Y->values, -X->values.data[i]);
        if(result) {
            return result;
        }
        ++Y->nnz;
        ++i;
    }
    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Y);
    /* Sort the indices */
    sptSparseTensorSortIndex(Y);
    return 0;
}

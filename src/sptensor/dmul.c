#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorDotMul(sptSparseTensor *Y, const sptSparseTensor *X) {
    size_t i, j;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        return -1;
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            return -1;
        }
    }
    /* Multiply elements one by one, assume indices are ordered */
    i = 0;
    j = 0;
    while(i < X->nnz && j < Y->nnz) {
        int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
        if(compare > 0) {
            ++j;
        } else if(compare < 0) {
            ++i;
        } else {
            Y->values.data[j] *= X->values.data[i];
            ++i;
            ++j;
        }
    }
    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Y);
    /* Sort the indices */
    sptSparseTensorSortIndex(Y);
    return 0;
}

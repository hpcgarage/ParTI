#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorDotMul(const sptSparseTensor *Y, const sptSparseTensor *X, sptSparseTensor *Z) {
    size_t i, j;
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        return -1;
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            fprintf(stderr, "SpTOL ERROR: Multiply tensors in different shapes.\n");
            return -1;
        }
    }

    sptNewSparseTensor(Z, X->nmodes, X->ndims);

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
            for(size_t mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Z->inds[mode], Y->inds[mode].data[j]);
                if(result) {
                    return result;
                }
            }
            result = sptAppendVector(&Z->values, Y->values.data[j]);
            if(result) {
                return result;
            }

            Y->values.data[Z->nnz] *= X->values.data[i];
            ++Z->nnz;
            ++i;
            ++j;
        }
    }
    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z);
    return 0;
}

#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorAdd(const sptSparseTensor *Y, const sptSparseTensor *X, sptSparseTensor *Z) {
    
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        return -1;
    }
    for(size_t i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            fprintf(stderr, "SpTOL ERROR: Adding tensors in different shapes.\n");
            return -1;
        }
    }

    sptNewSparseTensor(Z, X->nmodes, X->ndims);

    /* Add elements one by one, assume indices are ordered */
    size_t i, j;
    int result;
    i = 0;
    j = 0;
    while(i < X->nnz && j < Y->nnz) {
        int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
        printf("i: %lu, j: %lu, compare: %d\n", i,j,compare);

        if(compare > 0) {    // X(i) > Y(j)
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

            ++Z->nnz;
            ++j;
        } else if(compare < 0) {    // X(i) < Y(j)
            for(size_t mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Z->inds[mode], X->inds[mode].data[i]);
                if(result) {
                    return result;
                }
            }
            result = sptAppendVector(&Z->values, X->values.data[i]);
            if(result) {
                return result;
            }

            ++Z->nnz;
            ++i;
        } else {    // X(i) = Y(j)
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
            Z->values.data[Z->nnz] += X->values.data[i];

            ++Z->nnz;
            ++i;
            ++j;
        }
        printf("Z->values:\n");
        for(size_t ti=0; ti<Z->values.len; ++ti)
            printf("%lf ", Z->values.data[ti]);
        printf("\n");
    }
    /* Append remaining elements of X to Y */
    while(i < X->nnz) {
        size_t mode;
        int result;
        for(mode = 0; mode < X->nmodes; ++mode) {
            result = sptAppendSizeVector(&Z->inds[mode], X->inds[mode].data[i]);
            if(result) {
                return result;
            }
        }
        result = sptAppendVector(&Z->values, X->values.data[i]);
        if(result) {
            return result;
        }
        ++Z->nnz;
        ++i;
    }
    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z);
    return 0;
}

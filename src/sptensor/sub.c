#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorSub(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {

    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns Sub", "shape mismatch");
    }
    for(size_t i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns Sub", "shape mismatch");
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

        if(compare > 0) {
            for(size_t mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Z->inds[mode], Y->inds[mode].data[j]);
                spt_CheckError(result, "SpTns Sub", NULL);
            }
            result = sptAppendVector(&Z->values, Y->values.data[j]);
            spt_CheckError(result, "SpTns Sub", NULL);

            ++Z->nnz;
            ++j;
        } else if(compare < 0) {
            for(size_t mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Z->inds[mode], X->inds[mode].data[i]);
                spt_CheckError(result, "SpTns Sub", NULL);
            }
            result = sptAppendVector(&Z->values, -X->values.data[i]);
            spt_CheckError(result, "SpTns Sub", NULL);

            ++Z->nnz;
            ++i;
        } else {
            for(size_t mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Z->inds[mode], Y->inds[mode].data[j]);
                spt_CheckError(result, "SpTns Sub", NULL);
            }
            result = sptAppendVector(&Z->values, Y->values.data[j]);
            spt_CheckError(result, "SpTns Sub", NULL);

            Z->values.data[Z->nnz] -= X->values.data[i];
            ++Z->nnz;
            ++i;
            ++j;
        }
    }
    /* Append remaining elements of X to Y */
    while(i < X->nnz) {
        for(size_t mode = 0; mode < X->nmodes; ++mode) {
            result = sptAppendSizeVector(&Z->inds[mode], X->inds[mode].data[i]);
            spt_CheckError(result, "SpTns Sub", NULL);
        }
        result = sptAppendVector(&Z->values, -X->values.data[i]);
        spt_CheckError(result, "SpTns Sub", NULL);
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

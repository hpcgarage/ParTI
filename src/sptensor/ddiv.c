#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorDotDiv(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {
    size_t i, j;
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotDiv", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotDiv", "shape mismatch");
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
                spt_CheckError(result, "SpTns DotDiv", NULL);
            }
            result = sptAppendVector(&Z->values, Y->values.data[j]);
            spt_CheckError(result, "SpTns DotDiv", NULL);

            Y->values.data[Z->nnz] /= X->values.data[i];
            ++Z->nnz;
            ++i;
            ++j;
        }
    }

    /* Sort the indices */
    sptSparseTensorSortIndex(Z);
    return 0;
}

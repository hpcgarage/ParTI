#include <SpTOL.h>
#include "sptensor.h"

/**
 * Element wise multiply two sparse tensors, with exactly the same nonzero
 * distribution.
 * @param[out] Z the result of X*Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {
    size_t i, j;
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(Y->nnz != X->nnz) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "nonzero distribution mismatch");
    }
    size_t nnz = X->nnz;

    sptCopySparseTensor(Z, X);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(i=0; i< nnz; ++i)
        Z->values.data[i] = X->values.data[i] * Y->values.data[i];

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CPU  SpTns DotMul");
    sptFreeTimer(timer);

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z);
    return 0;
}

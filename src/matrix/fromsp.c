#include <SpTOL.h>
#include "../error/error.h"
#include <string.h>

/**
 * Convert sparse tensor to dense matrix
 *
 * @param dest pointer to an uninitialized matrix
 * @param src  pointer to a valid sparse tensor
 */
int sptSparseTensorToMatrix(sptMatrix *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    if(src->nmodes != 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns -> Mtx", "shape mismatch");
    }
    result = sptNewMatrix(dest, src->ndims[0], src->ndims[1]);
    spt_CheckError(result, "SpTns -> Mtx", NULL);
    memset(dest->values, 0, dest->nrows * dest->stride * sizeof (sptScalar));
    for(i = 0; i < src->nnz; ++i) {
        dest->values[src->inds[0].data[i] * dest->stride + src->inds[1].data[i]] = src->values.data[i];
    }
    return 0;
}

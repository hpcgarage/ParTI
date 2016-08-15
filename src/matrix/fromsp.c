#include <SpTOL.h>
#include <string.h>

int sptSparseTensorToMatrix(sptMatrix *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    if(src->nmodes != 2) {
        return -1;
    }
    result = sptNewMatrix(dest, src->ndims[0], src->ndims[1]);
    if(result != 0) {
        return result;
    }
    memset(dest->values, 0, dest->nrows * dest->stride * sizeof (sptScalar));
    for(i = 0; i < src->nnz; ++i) {
        dest->values[src->inds[0].data[i] * dest->stride + src->inds[1].data[i]] = src->values.data[i];
    }
    return 0;
}

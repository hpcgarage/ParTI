#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int sptNewMatrix(sptMatrix *mtx, size_t nrows, size_t ncols) {
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->stride = ((ncols-1)/8+1)*8;
#ifdef _ISOC11_SOURCE
    mtx->values = aligned_alloc(8 * sizeof (sptScalar), mtx->nrows * mtx->stride * sizeof (sptScalar));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int ok = posix_memalign((void **) &mtx->values, 8 * sizeof (sptScalar), mtx->nrows * mtx->stride * sizeof (sptScalar));
        if(!ok) {
            mtx->values = NULL;
        }
    }
#else
    mtx->values = malloc(mtx->nrows * mtx->stride * sizeof (sptScalar));
#endif
    if(!mtx->values) {
        return -1;
    }
    return 0;
}

int sptCopyMatrix(sptMatrix *dest, const sptMatrix *src) {
    int result = sptNewMatrix(dest, src->nrows, src->ncols);
    if(result) {
        return result;
    }
    assert(dest->stride == src->stride);
    memcpy(dest->values, src->values, dest->nrows * dest->stride * sizeof (sptScalar));
    return 0;
}

void sptFreeMatrix(sptMatrix *mtx) {
    free(mtx->values);
}

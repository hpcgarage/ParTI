#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int sptNewMatrix(sptMatrix *mtx, size_t nrows, size_t ncols) {
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->cap = nrows != 0 ? nrows : 1;
    mtx->stride = ((ncols-1)/8+1)*8;
#ifdef _ISOC11_SOURCE
    mtx->values = aligned_alloc(8 * sizeof (sptScalar), mtx->cap * mtx->stride * sizeof (sptScalar));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int ok = posix_memalign((void **) &mtx->values, 8 * sizeof (sptScalar), mtx->cap * mtx->stride * sizeof (sptScalar));
        if(!ok) {
            mtx->values = NULL;
        }
    }
#else
    mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptScalar));
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

int sptAppendMatrix(sptMatrix *mtx, const sptScalar values[]) {
    if(mtx->cap <= mtx->nrows) {
        size_t newcap = mtx->cap + mtx->cap/2;
        sptScalar *newdata;
#ifdef _ISOC11_SOURCE
        newdata = aligned_alloc(8 * sizeof (sptScalar), newcap * mtx->stride * sizeof (sptScalar));
#elif _POSIX_C_SOURCE >= 200112L
        {
            int ok = posix_memalign((void **) &newdata, 8 * sizeof (sptScalar), newcap * mtx->stride * sizeof (sptScalar));
            if(!ok) {
                newdata = NULL;
            }
        }
#else
        newdata = malloc(newcap * mtx->stride * sizeof (sptScalar));
#endif
        if(!newdata) {
            return -1;
        }
        memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (sptScalar));
        free(mtx->values);
        mtx->cap = newcap;
        mtx->values = newdata;
    }
    memcpy(&mtx->values[mtx->nrows * mtx->stride], values, mtx->ncols * sizeof (sptScalar));
    ++mtx->nrows;
    return 0;
}

void sptFreeMatrix(sptMatrix *mtx) {
    free(mtx->values);
}

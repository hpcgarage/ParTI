#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../error/error.h"

/**
 * Initialize a new dense matrix
 *
 * @param mtx   a valid pointer to an uninitialized sptMatrix variable,
 * @param nrows the number of rows
 * @param ncols the number of columns
 *
 * The memory layout of this dense matrix is a flat 2D array, with `ncols`
 * rounded up to multiples of 8
 */
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
    spt_CheckOSError(!mtx->values, "Matrix New");
    return 0;
}

/**
 * Fill a matrix with random number
 *
 * @param mtx   a pointer to a valid matrix
 * @param nrows fill the specified number of rows
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, size_t nrows, size_t ncols) {
  int result = sptNewMatrix(mtx, nrows, ncols);
  spt_CheckError(result, NULL, NULL);
  srand(time(NULL));
  for(size_t i=0; i<nrows; ++i)
    for(size_t j=0; j<ncols; ++j)
      mtx->values[i * mtx->stride + j] = (sptScalar) rand() / (sptScalar) RAND_MAX;
  return 0;
}

/**
 * Copy a dense matrix to an uninitialized dense matrix
 *
 * @param dest a pointer to an uninitialized dense matrix
 * @param src  a pointer to an existing valid dense matrix
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopyMatrix(sptMatrix *dest, const sptMatrix *src) {
    int result = sptNewMatrix(dest, src->nrows, src->ncols);
    if(result) {
        return result;
    }
    assert(dest->stride == src->stride);
    memcpy(dest->values, src->values, dest->nrows * dest->stride * sizeof (sptScalar));
    return 0;
}

/**
 * Add a row to the end of dense matrix
 *
 * @param mtx    a pointer to a valid matrix
 * @param values an array of data to be added
 */
int sptAppendMatrix(sptMatrix *mtx, const sptScalar values[]) {
    if(mtx->cap <= mtx->nrows) {
#ifndef MEMCHECK_MODE
        size_t newcap = mtx->cap + mtx->cap/2;
#else
        size_t newcap = mtx->nrows+1;
#endif
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
        spt_CheckOSError(!newdata, "Matrix Append");
        memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (sptScalar));
        free(mtx->values);
        mtx->cap = newcap;
        mtx->values = newdata;
    }
    if(values != NULL) {
        memcpy(&mtx->values[mtx->nrows * mtx->stride], values, mtx->ncols * sizeof (sptScalar));
    }
    ++mtx->nrows;
    return 0;
}

/**
 * Modify the number of rows in a dense matrix
 *
 * @param mtx     a pointer to a valid matrix
 * @param newsize the new number of rows `mtx` will have
 */
int sptResizeMatrix(sptMatrix *mtx, size_t newsize) {
    sptScalar *newdata;
#ifdef _ISOC11_SOURCE
    newdata = aligned_alloc(8 * sizeof (sptScalar), newsize * mtx->stride * sizeof (sptScalar));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int ok = posix_memalign((void **) &newdata, 8 * sizeof (sptScalar), newsize * mtx->stride * sizeof (sptScalar));
        if(!ok) {
            newdata = NULL;
        }
    }
#else
    newdata = malloc(newsize * mtx->stride * sizeof (sptScalar));
#endif
    if(!newdata) {
        return -1;
    }
    memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (sptScalar));
    free(mtx->values);
    mtx->nrows = newsize;
    mtx->cap = newsize;
    mtx->values = newdata;
    return 0;
}

/**
 * Release the memory buffer a dense matrix is holding
 *
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would becom uninitialized and should
 * not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    free(mtx->values);
}

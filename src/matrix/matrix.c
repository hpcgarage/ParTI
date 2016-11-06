/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../error/error.h"

/**
 * Initialize a new dense matrix
 *
 * @param mtx   a valid pointer to an uninitialized sptMatrix variable
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
    spt_CheckOSError(!mtx->values, "Mtx New");
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
  spt_CheckError(result, "Mtx Randomize", NULL);
  srand(time(NULL));
  for(size_t i=0; i<nrows; ++i)
    for(size_t j=0; j<ncols; ++j) {
      mtx->values[i * mtx->stride + j] = (sptScalar) rand() / (sptScalar) RAND_MAX;
    }
  return 0;
}

/**
 * Fill an existed dense matrix with a specified constant
 *
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptScalar const val) {
  for(size_t i=0; i<mtx->nrows; ++i)
    for(size_t j=0; j<mtx->ncols; ++j)
      mtx->values[i * mtx->stride + j] = val;
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
    spt_CheckError(result, "Mtx Copy", NULL);
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
        spt_CheckOSError(!newdata, "Mtx Append");
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
    spt_CheckOSError(!newdata, "Mtx Resize");
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
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    free(mtx->values);
}
/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
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
        int result = posix_memalign((void **) &mtx->values, 8 * sizeof (sptScalar), mtx->cap * mtx->stride * sizeof (sptScalar));
        if(result != 0) {
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
 * Build a matrix with random number
 *
 * @param mtx   a pointer to an uninitialized matrix
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
 * Fill an identity dense matrix
 *
 * @param mtx   a pointer to an uninitialized matrix
 * @param nrows fill the specified number of rows
 * @param ncols fill the specified number of columns
 *
 */
int sptIdentityMatrix(sptMatrix *mtx) {
  size_t const nrows = mtx->nrows;
  size_t const ncols = mtx->ncols;
  assert(nrows == ncols);
  for(size_t i=0; i<nrows; ++i)
    for(size_t j=0; j<ncols; ++j)
      mtx->values[i * mtx->stride + j] = 0;
  for(size_t i=0; i<nrows; ++i)
    mtx->values[i * mtx->stride + i] = 1;

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
            int result = posix_memalign((void **) &newdata, 8 * sizeof (sptScalar), newcap * mtx->stride * sizeof (sptScalar));
            if(result != 0) {
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
        int result = posix_memalign((void **) &newdata, 8 * sizeof (sptScalar), newsize * mtx->stride * sizeof (sptScalar));
        if(result != 0) {
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




/**
 * Initialize a new dense rank matrix
 *
 * @param mtx   a valid pointer to an uninitialized sptMatrix variable
 * @param nrows the number of rows
 * @param ncols the number of columns
 *
 * The memory layout of this dense matrix is a flat 2D array, with `ncols`
 * rounded up to multiples of 8
 */
int sptNewRankMatrix(sptRankMatrix *mtx, sptIndex nrows, sptElementIndex ncols) {
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->cap = nrows != 0 ? nrows : 1;
    mtx->stride = ((ncols-1)/8+1)*8;
#ifdef _ISOC11_SOURCE
    mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int result = posix_memalign((void **) &mtx->values, 8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
        if(result != 0) {
            mtx->values = NULL;
        }
    }
#else
    mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
    spt_CheckOSError(!mtx->values, "Mtx New");
    return 0;
}

/**
 * Fill an existed dense rank matrix with a specified constant
 *
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantRankMatrix(sptRankMatrix *mtx, sptValue const val) {
  for(sptIndex i=0; i<mtx->nrows; ++i)
    for(sptElementIndex j=0; j<mtx->ncols; ++j)
      mtx->values[i * mtx->stride + j] = val;
  return 0;
}


/**
 * Release the memory buffer a dense rank matrix is holding
 *
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeRankMatrix(sptRankMatrix *mtx) {
    free(mtx->values);
}




int sptMatrixDotMul(sptMatrix const * A, sptMatrix const * B, sptMatrix const * C)
{
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    size_t stride = A->stride;
    assert(nrows == B->nrows && nrows == C->nrows);
    assert(ncols == B->ncols && ncols == C->ncols);
    assert(stride == B->stride && stride == C->stride);

    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            C->values[i*stride+j] = A->values[i*stride+j] * B->values[i*stride+j];
        }
    }

    return 0;
}


int sptMatrixDotMulSeq(size_t const mode, size_t const nmodes, sptMatrix ** mats)
{
    size_t const nrows = mats[0]->nrows;
    size_t const ncols = mats[0]->ncols;
    size_t const stride = mats[0]->stride;
    // printf("stride: %lu\n", stride);
    for(size_t m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    sptScalar * ovals = mats[nmodes]->values;
    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            ovals[i * stride + j] = 1;
        }
    }

    for(size_t m=1; m < nmodes; ++m) {
        size_t const pm = (mode + m) % nmodes;
        sptScalar const * vals = mats[pm]->values;
        for(size_t i=0; i < nrows; ++i) {
            for(size_t j=0; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }
    
    return 0;
}


int sptMatrixDotMulSeqCol(size_t const mode, size_t const nmodes, sptMatrix ** mats)
{
    size_t const nrows = mats[0]->nrows;
    size_t const ncols = mats[0]->ncols;
    size_t const stride = mats[0]->stride;
    // printf("stride: %lu\n", stride);
    for(size_t m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    sptScalar * ovals = mats[nmodes]->values;
    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            ovals[j * stride + i] = 1;
        }
    }

    for(size_t m=1; m < nmodes; ++m) {
        size_t const pm = (mode + m) % nmodes;
        sptScalar const * vals = mats[pm]->values;
        for(size_t i=0; i < nrows; ++i) {
            for(size_t j=0; j < ncols; ++j) {
                ovals[j * stride + i] *= vals[j * stride + i];
            }
        }
    }
    
    return 0;
}



int sptOmpMatrixDotMulSeq(size_t const mode, size_t const nmodes, sptMatrix ** mats)
{
    size_t const nrows = mats[0]->nrows;
    size_t const ncols = mats[0]->ncols;
    size_t const stride = mats[0]->stride;
    // printf("stride: %lu\n", stride);
    for(size_t m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    sptScalar * ovals = mats[nmodes]->values;
    #pragma omp parallel for
    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            ovals[i * stride + j] = 1;
        }
    }

    for(size_t m=1; m < nmodes; ++m) {
        size_t const pm = (mode + m) % nmodes;
        // printf("pm: %lu\n", pm);
        sptScalar const * vals = mats[pm]->values;
        #pragma omp parallel for
        for(size_t i=0; i < nrows; ++i) {
            for(size_t j=0; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }
    
    return 0;
}



// Row-major
int sptMatrix2Norm(sptMatrix * const A, sptScalar * const lambda)
{
    size_t const nrows = A->nrows;
    size_t const ncols = A->ncols;
    size_t const stride = A->stride;
    sptScalar * const vals = A->values;

    for(size_t j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            lambda[j] += vals[i*stride + j] * vals[i*stride + j];
        }
    }
    for(size_t j=0; j < ncols; ++j) {
        lambda[j] = sqrt(lambda[j]);
    }

    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            vals[i*stride + j] /= lambda[j];
        }
    }

    return 0;
}



int sptOmpMatrix2Norm(sptMatrix * const A, sptScalar * const lambda)
{
    size_t const nrows = A->nrows;
    size_t const ncols = A->ncols;
    size_t const stride = A->stride;
    sptScalar * const vals = A->values;

    for(size_t j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        sptScalar * loc_lambda = (sptScalar *)malloc(ncols * nthreads * sizeof(sptScalar));
        for(size_t j=0; j < ncols * nthreads; ++j)
            loc_lambda[j] = 0;
        for(size_t j=0; j < ncols; ++j)
            lambda[j] = 0;

        #pragma omp for
        for(size_t i=0; i < nrows; ++i) {
            for(size_t j=0; j < ncols; ++j) {
                loc_lambda[tid * ncols + j] += vals[i*stride + j] * vals[i*stride + j];
            }
        }

        for(int i=0; i<nthreads; ++i) {
            for(size_t j=0; j < ncols; ++j) {
                lambda[j] += loc_lambda[i*ncols + j];
            }
        }

        #pragma omp for
        for(size_t j=0; j < ncols; ++j) {
            lambda[j] = sqrt(lambda[j]);
        }

        #pragma omp for
        for(size_t i=0; i < nrows; ++i) {
            for(size_t j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

    }   /* end parallel pragma */

    return 0;
}


// Row-major
int sptMatrixMaxNorm(sptMatrix * const A, sptScalar * const lambda)
{
    size_t const nrows = A->nrows;
    size_t const ncols = A->ncols;
    size_t const stride = A->stride;
    sptScalar * const vals = A->values;

    for(size_t j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            if(vals[i*stride + j] > lambda[j])
                lambda[j] = vals[i*stride + j];
        }
    }
    for(size_t j=0; j < ncols; ++j) {
        if(lambda[j] < 1)
            lambda[j] = 1;
    }

    for(size_t i=0; i < nrows; ++i) {
        for(size_t j=0; j < ncols; ++j) {
            vals[i*stride + j] /= lambda[j];
        }
    }

    return 0;
}


void GetFinalLambda(
  size_t const rank,
  size_t const nmodes,
  sptMatrix ** mats,
  sptScalar * const lambda)
{
  sptScalar * tmp_lambda =  (sptScalar *) malloc(rank * sizeof(*tmp_lambda));

  for(size_t m=0; m < nmodes; ++m) {
    sptMatrix2Norm(mats[m], tmp_lambda);
    for(size_t r=0; r < rank; ++r) {
      lambda[r] *= tmp_lambda[r];
    }
  }

  free(tmp_lambda);
}
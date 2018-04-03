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
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
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
 * Build a matrix with random number
 *
 * @param mtx   a pointer to an uninitialized matrix
 * @param nrows fill the specified number of rows
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
  srand(time(NULL));
  for(sptIndex i=0; i<nrows; ++i)
    for(sptIndex j=0; j<ncols; ++j) {
      mtx->values[i * mtx->stride + j] = sptRandomValue();
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
  sptIndex const nrows = mtx->nrows;
  sptIndex const ncols = mtx->ncols;
  assert(nrows == ncols);
  for(sptIndex i=0; i<nrows; ++i)
    for(sptIndex j=0; j<ncols; ++j)
      mtx->values[i * mtx->stride + j] = 0;
  for(sptIndex i=0; i<nrows; ++i)
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
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
  for(sptIndex i=0; i<mtx->nrows; ++i)
    for(sptIndex j=0; j<mtx->ncols; ++j)
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
    memcpy(dest->values, src->values, dest->nrows * dest->stride * sizeof (sptValue));
    return 0;
}

/**
 * Add a row to the end of dense matrix
 *
 * @param mtx    a pointer to a valid matrix
 * @param values an array of data to be added
 */
int sptAppendMatrix(sptMatrix *mtx, const sptValue values[]) {
    if(mtx->cap <= mtx->nrows) {
#ifndef MEMCHECK_MODE
        sptIndex newcap = mtx->cap + mtx->cap/2;
#else
        sptIndex newcap = mtx->nrows+1;
#endif
        sptValue *newdata;
#ifdef _ISOC11_SOURCE
        newdata = aligned_alloc(8 * sizeof (sptValue), newcap * mtx->stride * sizeof (sptValue));
#elif _POSIX_C_SOURCE >= 200112L
        {
            int result = posix_memalign((void **) &newdata, 8 * sizeof (sptValue), newcap * mtx->stride * sizeof (sptValue));
            if(result != 0) {
                newdata = NULL;
            }
        }
#else
        newdata = malloc(newcap * mtx->stride * sizeof (sptValue));
#endif
        spt_CheckOSError(!newdata, "Mtx Append");
        memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (sptValue));
        free(mtx->values);
        mtx->cap = newcap;
        mtx->values = newdata;
    }
    if(values != NULL) {
        memcpy(&mtx->values[mtx->nrows * mtx->stride], values, mtx->ncols * sizeof (sptValue));
    }
    ++ mtx->nrows;
    return 0;
}

/**
 * Modify the number of rows in a dense matrix
 *
 * @param mtx     a pointer to a valid matrix
 * @param new_nrows the new number of rows `mtx` will have
 */
int sptResizeMatrix(sptMatrix *mtx, sptIndex const new_nrows) {
    sptValue *newdata;
#ifdef _ISOC11_SOURCE
    newdata = aligned_alloc(8 * sizeof (sptValue), new_nrows * mtx->stride * sizeof (sptValue));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int result = posix_memalign((void **) &newdata, 8 * sizeof (sptValue), new_nrows * mtx->stride * sizeof (sptValue));
        if(result != 0) {
            newdata = NULL;
        }
    }
#else
    newdata = malloc(new_nrows * mtx->stride * sizeof (sptValue));
#endif
    spt_CheckOSError(!newdata, "Mtx Resize");
    memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (sptValue));
    free(mtx->values);
    mtx->nrows = new_nrows;
    mtx->cap = new_nrows;
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
    mtx->nrows = 0;
    mtx->ncols = 0;
    mtx->cap = 0;
    mtx->stride = 0;
}


/**** sptMatrix Operations ****/

int sptMatrixDotMul(sptMatrix const * A, sptMatrix const * B, sptMatrix const * C)
{
    sptIndex nrows = A->nrows;
    sptIndex ncols = A->ncols;
    sptIndex stride = A->stride;
    assert(nrows == B->nrows && nrows == C->nrows);
    assert(ncols == B->ncols && ncols == C->ncols);
    assert(stride == B->stride && stride == C->stride);

    for(sptIndex i=0; i < nrows; ++i) {
        for(sptIndex j=0; j < ncols; ++j) {
            C->values[i*stride+j] = A->values[i*stride+j] * B->values[i*stride+j];
        }
    }

    return 0;
}


int sptMatrixDotMulSeq(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats)
{
    sptIndex const nrows = mats[0]->nrows;
    sptIndex const ncols = mats[0]->ncols;
    sptIndex const stride = mats[0]->stride;

    for(sptIndex m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    sptValue * ovals = mats[nmodes]->values;
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(sptIndex i=0; i < nrows; ++i) {
        for(sptIndex j=0; j < ncols; ++j) {
            ovals[i * stride + j] = 1;
        }
    }

    for(sptIndex m=1; m < nmodes; ++m) {
        sptIndex const pm = (mode + m) % nmodes;
        sptValue const * vals = mats[pm]->values;
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(sptIndex i=0; i < nrows; ++i) {
            for(sptIndex j=0; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }
    
    return 0;
}


int sptMatrixDotMulSeqCol(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats)
{
    sptIndex const nrows = mats[0]->nrows;
    sptIndex const ncols = mats[0]->ncols;
    sptIndex const stride = mats[0]->stride;
    // printf("stride: %lu\n", stride);
    for(sptIndex m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    sptValue * ovals = mats[nmodes]->values;
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(sptIndex j=0; j < ncols; ++j) {
        for(sptIndex i=0; i < nrows; ++i) {
            ovals[j * stride + i] = 1;
        }
    }


    for(sptIndex m=1; m < nmodes; ++m) {
        sptIndex const pm = (mode + m) % nmodes;
        sptValue const * vals = mats[pm]->values;
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(sptIndex j=0; j < ncols; ++j) {
            for(sptIndex i=0; i < nrows; ++i) {
                ovals[j * stride + i] *= vals[j * stride + i];
            }
        }
    }
    
    return 0;
}


/* mats (aTa) only stores upper triangle elements. */
int sptMatrixDotMulSeqTriangle(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats)
{
    sptIndex const nrows = mats[0]->nrows;
    sptIndex const ncols = mats[0]->ncols;
    sptIndex const stride = mats[0]->stride;
    for(sptIndex m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
    }

    sptValue * ovals = mats[nmodes]->values;
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(sptIndex i=0; i < nrows; ++i) {
        for(sptIndex j=0; j < ncols; ++j) {
            ovals[j * stride + i] = 1.0;
        }
    }


    for(sptIndex m=1; m < nmodes; ++m) {
        sptIndex const pm = (mode + m) % nmodes;
        sptValue const * vals = mats[pm]->values;
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
        for(sptIndex i=0; i < nrows; ++i) {
            for(sptIndex j=i; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }

    /* Copy upper triangle to lower part */
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(sptIndex i=0; i < nrows; ++i) {
        for(sptIndex j=0; j < i; ++j) {
            ovals[i * stride + j] = ovals[j * stride + i];
        }
    }
    
    return 0;
}


// Row-major
int sptMatrix2Norm(sptMatrix * const A, sptValue * const lambda)
{
    sptIndex const nrows = A->nrows;
    sptIndex const ncols = A->ncols;
    sptIndex const stride = A->stride;
    sptValue * const vals = A->values;
    sptValue * buffer_lambda;

#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(sptIndex j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

#ifdef PARTI_USE_OPENMP
    #pragma omp parallel
    {
        int const nthreads = omp_get_num_threads();
        #pragma omp master
        {
            buffer_lambda = (sptValue *)malloc(nthreads * ncols * sizeof(sptValue));
            for(sptNnzIndex j=0; j < nthreads * ncols; ++j)
                buffer_lambda[j] = 0.0;
        }
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        sptValue * loc_lambda = buffer_lambda + tid * ncols;

        #pragma omp for
        for(sptIndex i=0; i < nrows; ++i) {
            for(sptIndex j=0; j < ncols; ++j) {
                loc_lambda[j] += vals[i*stride + j] * vals[i*stride + j];
            }
        }

        #pragma omp for
        for(sptIndex j=0; j < ncols; ++j) {
            for(int i=0; i < nthreads; ++i) {
                lambda[j] += buffer_lambda[i*ncols + j];
            }
        }
    }   /* end parallel pragma */

#else

    for(sptIndex i=0; i < nrows; ++i) {
        for(sptIndex j=0; j < ncols; ++j) {
            lambda[j] += vals[i*stride + j] * vals[i*stride + j];
        }
    }

#endif

#ifdef PARTI_USE_OPENMP
        #pragma omp for
#endif
        for(sptIndex j=0; j < ncols; ++j) {
            lambda[j] = sqrt(lambda[j]);
        }

#ifdef PARTI_USE_OPENMP
        #pragma omp for
#endif
        for(sptIndex i=0; i < nrows; ++i) {
            for(sptIndex j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

    
#ifdef PARTI_USE_OPENMP
    free(buffer_lambda);
#endif

    return 0;
}

// Row-major
int sptMatrixMaxNorm(sptMatrix * const A, sptValue * const lambda)
{
    sptIndex const nrows = A->nrows;
    sptIndex const ncols = A->ncols;
    sptIndex const stride = A->stride;
    sptValue * const vals = A->values;
    sptValue * buffer_lambda;

#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(sptIndex j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

#ifdef PARTI_USE_OPENMP
    #pragma omp parallel
    {
        int const nthreads = omp_get_num_threads();
        #pragma omp master
        {
            buffer_lambda = (sptValue *)malloc(nthreads * ncols * sizeof(sptValue));
            for(sptNnzIndex j=0; j < nthreads * ncols; ++j)
                buffer_lambda[j] = 0.0;
        }
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        sptValue * loc_lambda = buffer_lambda + tid * ncols;

        #pragma omp for
        for(sptIndex i=0; i < nrows; ++i) {
            for(sptIndex j=0; j < ncols; ++j) {
                if(vals[i*stride + j] > loc_lambda[j])
                    loc_lambda[j] = vals[i*stride + j];
            }
        }

        #pragma omp for
        for(sptIndex j=0; j < ncols; ++j) {
            for(int i=0; i < nthreads; ++i) {
                if(buffer_lambda[i*ncols + j] > lambda[j])
                    lambda[j] = buffer_lambda[i*ncols + j];
            }
        }
    }   /* end parallel pragma */

#else
    for(sptIndex i=0; i < nrows; ++i) {
        for(sptIndex j=0; j < ncols; ++j) {
            if(vals[i*stride + j] > lambda[j])
                lambda[j] = vals[i*stride + j];
        }
    }
#endif

#ifdef PARTI_USE_OPENMP
        #pragma omp for
#endif
        for(sptIndex j=0; j < ncols; ++j) {
            if(lambda[j] < 1)
                lambda[j] = 1;
        }

#ifdef PARTI_USE_OPENMP
        #pragma omp for
#endif
        for(sptIndex i=0; i < nrows; ++i) {
            for(sptIndex j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

#ifdef PARTI_USE_OPENMP
    free(buffer_lambda);
#endif

    return 0;
}


void GetFinalLambda(
  sptIndex const rank,
  sptIndex const nmodes,
  sptMatrix ** mats,
  sptValue * const lambda)
{
  sptValue * tmp_lambda =  (sptValue *) malloc(rank * sizeof(*tmp_lambda));

  for(sptIndex m=0; m < nmodes; ++m) {   
    sptMatrix2Norm(mats[m], tmp_lambda);
    for(sptIndex r=0; r < rank; ++r) {
      lambda[r] *= tmp_lambda[r];
    }
  }

  free(tmp_lambda);
}
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
#include "../error/error.h"

/**
 * Initialize a new sparse matrix
 *
 * @param mtx   a valid pointer to an uninitialized sptSparseMatrix variable
 * @param nrows the number of rows
 * @param ncols the number of columns
 */
int sptNewSparseMatrix(sptSparseMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    int result;
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->nnz = 0;
    result = sptNewIndexVector(&mtx->rowind, 0, 0);
    spt_CheckError(result, "SpMtx New", NULL);
    result = sptNewIndexVector(&mtx->colind, 0, 0);
    spt_CheckError(result, "SpMtx New", NULL);
    result = sptNewValueVector(&mtx->values, 0, 0);
    spt_CheckError(result, "SpMtx New", NULL);
    return 0;
}

/**
 * Copy a sparse matrix to an uninitialized sparse matrix
 *
 * @param dest a pointer to an uninitialized sparse matrix
 * @param src  a pointer to an existing valid sparse matrix
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopySparseMatrix(sptSparseMatrix *dest, const sptSparseMatrix *src) {
    int result;
    dest->nrows = src->nrows;
    dest->ncols = src->ncols;
    dest->nnz = src->nnz;
    result = sptCopyIndexVector(&dest->rowind, &src->rowind, 1);
    spt_CheckError(result, "SpMtx Copy", NULL);
    result = sptCopyIndexVector(&dest->colind, &src->colind, 1);
    spt_CheckError(result, "SpMtx Copy", NULL);
    result = sptCopyValueVector(&dest->values, &src->values, 1);
    spt_CheckError(result, "SpMtx Copy", NULL);
    return 0;
}

/**
 * Release the memory buffer a sparse matrix is holding
 *
 * @param mtx a pointer to a valid sparse matrix
 *
 * By using `sptFreeSparseMatrix`, a valid sparse matrix would become
 * uninitialized and should not be used anymore prior to another initialization
 */
void sptFreeSparseMatrix(sptSparseMatrix *mtx) {
    sptFreeIndexVector(&mtx->rowind);
    sptFreeIndexVector(&mtx->colind);
    sptFreeValueVector(&mtx->values);
    mtx->nrows = 0;
    mtx->ncols = 0;
    mtx->nnz = 0;
}

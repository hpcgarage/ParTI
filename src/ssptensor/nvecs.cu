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
#include "ssptensor.h"
#include <math.h>
#include <cusparse.h>

int sptSemiSparseTensorNvecs(
    sptSparseTensor           *u,
    const sptSemiSparseTensor *t,
    size_t                    n,
    size_t                    r
) {
    sptSemiSparseTensor tn;
    spt_SemiSparseTensorSetMode(&tn, t, n);
    float *tnt_val = new float[tn.nnz * tn.ndims[tn.mode]];
    int *tnt_rowptr = new int[tn.nnz + 1];
    int *tnt_colind = new int[tn.nnz * tn.ndims[tn.mode]];
    spt_SemiSparseTensorToSparseMatrixCSR(tnt_val, tnt_rowptr, tnt_colind, &tn);

    float *y_val;
    size_t *y_rowptr;
    size_t *y_colind;

    cusparseMatDescr_t matrix_descriptor;
    cusparseCreateMatDescr(&matrix_descriptor);
    cusparseSetMatType(matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(matrix_descriptor, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(matrix_descriptor, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

    cusparseXcsrgemmNnz(
        /* handle */ NULL, // TODO
        /* transA */ CUSPARSE_OPERATION_TRANSPOSE,
        /* transB */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* m */ tn.nnz,
        /* n */ tn.nnz,
        /* k */ tn.ndims[tn.mode],
        /* descrA */ matrix_descriptor,
        /* nnzA */ tn.nnz * tn.ndims[tn.mode],
        /* csrRowPtrA */ tnt_rowptr,
        /* csrColIndA */ tnt_colind,
        /* descrB */ matrix_descriptor,
        /* nnzB */ tn.nnz * tn.ndims[tn.mode],
        /* csrRowPtrB */ tnt_rowptr,
        /* csrColIndB */ tnt_colind,
        /* descrC */ matrix_descriptor,
        /* csrRowPtrC */ NULL, // TODO
        /* nnzTotalDevHostPtr */ NULL // TODO
    );

    // TODO

    cusparseDestroyMatDescr(matrix_descriptor);
    delete[] tnt_colind;
    delete[] tnt_rowptr;
    delete[] tnt_val;
    sptFreeSemiSparseTensor(&tn);

    return 0;
}

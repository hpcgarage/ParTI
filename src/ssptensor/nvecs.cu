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
#include "../cudawrap.h"

int sptSemiSparseTensorNvecs(
    sptSparseTensor           *u,
    const sptSemiSparseTensor *t,
    size_t                    n,
    size_t                    r
) {
    int result = 0;
    cusparseHandle_t cusparse_handle;
    result = spt_cusparseCreate(&cusparse_handle);
    spt_CheckCudaError(result, "NVECS");

    sptSemiSparseTensor tn;
    spt_SemiSparseTensorSetMode(&tn, t, n);

    size_t tnt_width = tn.nnz;
    size_t tnt_height = tn.ndims[tn.mode];

    float *tnt_val = new float[tnt_width * tnt_height];
    int *tnt_rowptr = new int[tnt_height + 1];
    int *tnt_colind = new int[tnt_width * tnt_height];
    result = spt_SemiSparseTensorToSparseMatrixCSR(tnt_val, tnt_rowptr, tnt_colind, &tn);
    spt_CheckCudaError(result, "NVECS");
    sptFreeSemiSparseTensor(&tn);

    float *dev_tnt_val;
    int *dev_tnt_rowptr;
    int *dev_tnt_colind;
    result = sptCudaDuplicateMemory(&dev_tnt_val, tnt_val, tnt_width * tnt_height * sizeof *tnt_val, cudaMemcpyHostToDevice);
    spt_CheckCudaError(result, "NVECS");
    result = sptCudaDuplicateMemory(&dev_tnt_rowptr, tnt_rowptr, (tnt_height + 1) * sizeof *tnt_rowptr, cudaMemcpyHostToDevice);
    spt_CheckCudaError(result, "NVECS");
    result = sptCudaDuplicateMemory(&dev_tnt_colind, tnt_colind, tnt_width * tnt_height * sizeof *tnt_colind, cudaMemcpyHostToDevice);
    spt_CheckCudaError(result, "NVECS");

    delete[] tnt_colind;
    delete[] tnt_rowptr;
    delete[] tnt_val;

    float *dev_y_val;
    int *dev_y_rowptr;
    int *dev_y_colind;
    int y_nnz;

    result = cudaMalloc((int **) &dev_y_rowptr, (tnt_height + 1) * sizeof *dev_y_rowptr);
    spt_CheckCudaError(result, "NVECS");

    cusparseMatDescr_t matrix_descriptor;
    cusparseCreateMatDescr(&matrix_descriptor);
    cusparseSetMatType(matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(matrix_descriptor, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(matrix_descriptor, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

    result = cusparseXcsrgemmNnz(
        /* handle */ cusparse_handle,
        /* transA */ CUSPARSE_OPERATION_TRANSPOSE,
        /* transB */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* m */ tnt_width,
        /* n */ tnt_width,
        /* k */ tnt_height,
        /* descrA */ matrix_descriptor,
        /* nnzA */ tnt_width * tnt_height,
        /* csrRowPtrA */ dev_tnt_rowptr,
        /* csrColIndA */ dev_tnt_colind,
        /* descrB */ matrix_descriptor,
        /* nnzB */ tnt_width * tnt_height,
        /* csrRowPtrB */ dev_tnt_rowptr,
        /* csrColIndB */ dev_tnt_colind,
        /* descrC */ matrix_descriptor,
        /* csrRowPtrC */ dev_y_rowptr,
        /* nnzTotalDevHostPtr */ &y_nnz
    );
    spt_CheckCudaError(result, "NVECS");

    result = cudaMalloc((float **) &dev_y_val, y_nnz * sizeof *dev_y_val);
    spt_CheckCudaError(result, "NVECS");
    result = cudaMalloc((int **) &dev_y_colind, y_nnz * sizeof *dev_y_colind);
    spt_CheckCudaError(result, "NVECS");

    result = cusparseScsrgemm(
        /* handle */ cusparse_handle,
        /* transA */ CUSPARSE_OPERATION_TRANSPOSE,
        /* transB */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* m */ tnt_width,
        /* n */ tnt_width,
        /* k */ tnt_height,
        /* descrA */ matrix_descriptor,
        /* nnzA */ tnt_width * tnt_height,
        /* csrValA */ dev_tnt_val,
        /* csrRowPtrA */ dev_tnt_rowptr,
        /* csrColIndA */ dev_tnt_colind,
        /* descrB */ matrix_descriptor,
        /* nnzB */ tnt_width * tnt_height,
        /* csrValB */ dev_tnt_val,
        /* csrRowPtrB */ dev_tnt_rowptr,
        /* csrColIndB */ dev_tnt_colind,
        /* descrC */ matrix_descriptor,
        /* csrValC */ dev_y_val,
        /* csrRowPtrC */ dev_y_rowptr,
        /* csrColIndC */ dev_y_colind
    );
    spt_CheckCudaError(result, "NVECS");

    cudaFree(dev_tnt_colind);
    cudaFree(dev_tnt_rowptr);
    cudaFree(dev_tnt_val);

    cusparseDestroyMatDescr(matrix_descriptor);

    return 0;
}

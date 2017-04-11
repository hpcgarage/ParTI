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


/**
 * Create a sparse matrix representation of an semi-sparse tensor `tsr`.
 * The mode specified in `tsr->modemode` map to the column of the matrix,
 * and the remaining modes (in ascending order) map to the rows.
 *
 * @parameter[out] csrVal     preallocated length = tsr->nnz * tsr->ndims[tsr->mode]
 * @parameter[out] csrRowPtr  preallocated length = tsr->nnz+1
 * @parameter[out] csrColInd  preallocated length = tsr->nnz * tsr->ndims[tsr->mode]
 */
int spt_SemiSparseTensorToSparseMatrixCSR(
    sptScalar                  *csrVal,
    int                        *csrRowPtr,
    int                        *csrColInd,
    const sptSemiSparseTensor  *tsr
) {
    /*
        For tensor size [X, Y, Z], col_dim = 1,
        this function will output a matrix [X * Z, Y],
        where [a, b, c] will map to [a * X + c, b].
    */

    const size_t stride = tsr->stride;
    const size_t ncols = tsr->ndims[tsr->mode];
    for(size_t row = 0; row < tsr->nnz; ++row) {
        for(size_t col = 0; col < ncols; ++col) {
            csrVal[row * ncols + col] = tsr->values.values[row * stride + col];
        }
    }

    for(size_t row = 0; row <= tsr->nnz; ++row) {
        csrRowPtr[row] = row * ncols;
    }

    for(size_t row = 0; row < tsr->nnz; ++row) {
        for(size_t col = 0; col < ncols; ++col) {
            csrColInd[row * ncols + col] = col;
        }
    }

    return 0;
}

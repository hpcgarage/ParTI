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
#include <stdlib.h>

/**
 * Semi sparse tensor times a dense matrix (SspTTM)
 * @param[out] Y    the result of X*U, should be uninitialized
 * @param[in]  X    the semi sparse tensor input X
 * @param[in]  U    the dense matrix input U
 * @param      mode the mode on which the multiplication is done on
 */
int sptSemiSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    const sptSemiSparseTensor *X,
    const sptMatrix *U,
    sptIndex mode
) {
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndex i;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    if(!ind_buf) {
        return -1;
    }
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    // jli: use pre-processing to allocate Y size outside this function.
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    if(result) {
        return result;
    }
    for(m = 0; m < Y->nmodes; ++m) {
        if(m != mode) {
            sptFreeIndexVector(&Y->inds[m]);
            result = sptCopyIndexVector(&Y->inds[m], &X->inds[m]);
            if(result != 0) {
                return result;
            }
        }
    }
    result = sptResizeMatrix(&Y->values, X->nnz);
    if(result != 0) {
        return result;
    }
    Y->nnz = X->nnz;
    memset(Y->values.values, 0, Y->nnz * Y->stride * sizeof (sptValue));
    for(i = 0; i < X->nnz; ++i) {
        sptIndex r, k;
        for(k = 0; k < U->ncols; ++k) {
            for(r = 0; r < U->nrows; ++r) {
                Y->values.values[i*Y->stride + k] += X->values.values[i*X->stride + r] * U->values[r*U->stride + k];
            }
        }
    }
    return 0;
}

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
#include <stdlib.h>
#include "sptensor.h"

/**
 * Sparse tensor times a dense matrix (SpTTM)
 * @param[out] Y    the result of X*U, should be uninitialized
 * @param[in]  X    the sparse tensor input X
 * @param[in]  U    the dense matrix input U
 * @param      mode the mode on which the multiplication is done on
 *
 * This function will sort Y with `sptSparseTensorSortIndexAtMode`
 * automatically, this operation can be undone with `sptSparseTensorSortIndex`
 * if you need to access raw data.
 * Anyway, you do not have to take this side-effect into consideration if you
 * do not need to access raw data.
 */
int sptSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode) {
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndex i;
    sptNnzIndexVector fiberidx;
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * Mtx", "shape mismatch");
    }
    sptSparseTensorSortIndexAtMode(X, mode, 0, 1);
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "CPU  SpTns * Mtx");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    // jli: use pre-processing to allocate Y size outside this function.
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    spt_CheckError(result, "CPU  SpTns * Mtx", NULL);
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(i = 0; i < Y->nnz; ++i) {
        sptNnzIndex inz_begin = fiberidx.data[i];
        sptNnzIndex inz_end = fiberidx.data[i+1];
        // jli: exchange the two loops
        for(sptNnzIndex j = inz_begin; j < inz_end; ++j) {
            sptIndex r = X->inds[mode].data[j];
            for(sptIndex k = 0; k < U->ncols; ++k) {
                Y->values.values[i*Y->stride + k] += X->values.data[j] * U->values[r*U->stride + k];
            }
        }
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CPU  SpTns * Mtx");
    sptFreeTimer(timer);

    sptFreeNnzIndexVector(&fiberidx);
    return 0;
}

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

int sptOmpSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode) {
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndexVector fiberidx;
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  SpTns * Mtx", "shape mismatch");
    }
    sptSparseTensorSortIndexAtMode(X, mode, 0, 1);
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "OMP  SpTns * Mtx");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    spt_CheckError(result, "OMP  SpTns * Mtx", NULL);
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    #pragma omp parallel for
    for(sptNnzIndex i = 0; i < Y->nnz; ++i) {
        sptNnzIndex inz_begin = fiberidx.data[i];
        sptNnzIndex inz_end = fiberidx.data[i+1];
        // jli: exchange two loops
        for(sptNnzIndex j = inz_begin; j < inz_end; ++j) {
            sptIndex r = X->inds[mode].data[j];
            for(sptIndex k = 0; k < U->ncols; ++k) {
                Y->values.values[i*Y->stride + k] += X->values.data[j] * U->values[r*U->stride + k];
            }
        }
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "OMP  SpTns * Mtx");
    sptFreeTimer(timer);

    sptFreeNnzIndexVector(&fiberidx);
    return 0;
}

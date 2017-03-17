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

int sptOmpSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, size_t mode) {
    int result;
    size_t *ind_buf;
    size_t m, i;
    sptSizeVector fiberidx;
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  SpTns * Mtx", "shape mismatch");
    }
    sptSparseTensorSortIndexAtMode(X, mode, 0);
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "OMP  SpTns * Mtx");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    // jli: use pre-processing to allocate Y size outside this function.
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    spt_CheckError(result, "OMP  SpTns * Mtx", NULL);
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    #pragma omp parallel for
    for(i = 0; i < Y->nnz; ++i) {
        size_t inz_begin = fiberidx.data[i];
        size_t inz_end = fiberidx.data[i+1];
        size_t j, k;
        // jli: exchange two loops
        for(j = inz_begin; j < inz_end; ++j) {
            size_t r = X->inds[mode].data[j];
            for(k = 0; k < U->ncols; ++k) {
                Y->values.values[i*Y->stride + k] += X->values.data[j] * U->values[r*U->stride + k];
            }
        }
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "OMP  SpTns * Mtx");
    sptFreeTimer(timer);

    sptFreeSizeVector(&fiberidx);
    return 0;
}

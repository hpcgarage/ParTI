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
 * Sparse tensor times a vector (SpTTV)
 */
int sptSparseTensorMulVector(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptVector *V, size_t mode) {
    int result;
    size_t *ind_buf;
    size_t m, i;
    sptSizeVector fiberidx;
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * Vec", "shape mismatch");
    }
    if(X->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * Vec", "shape mismatch");
    }
    sptSparseTensorSortIndexAtMode(X, mode, 0);
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "CPU  SpTns * Vec");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = 1;
    // jli: use pre-processing to allocate Y size outside this function.
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    spt_CheckError(result, "CPU  SpTns * Vec", NULL);
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(i = 0; i < Y->nnz; ++i) {
        size_t inz_begin = fiberidx.data[i];
        size_t inz_end = fiberidx.data[i+1];
        size_t j;
        // jli: exchange the two loops
        for(j = inz_begin; j < inz_end; ++j) {
            size_t r = X->inds[mode].data[j];
            Y->values.values[i*Y->stride] += X->values.data[j] * V->data[r];
        }
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CPU  SpTns * Vec");
    sptFreeTimer(timer);

    sptFreeSizeVector(&fiberidx);
    return 0;
}

/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <SpTOL.h>
#include "ssptensor.h"
#include <string.h>

static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]);

/**
 * Merge fibers with identical indices of an invalid semi sparse tensor, making it valid
 * @param tsr the semi sparse tensor to operate on
 */
int spt_SemiSparseTensorMergeValues(sptSemiSparseTensor *tsr) {
    int result;
    size_t i;
    sptSizeVector collided;
    sptScalar *buffer;

    if(tsr->nnz == 0) {
        return 0;
    }

    buffer = malloc(tsr->stride * sizeof (sptScalar));
    spt_CheckOSError(!buffer, "SspTns Merge");

    result = sptNewSizeVector(&collided, 0, 0);
    if(result) {
        free(buffer);
        spt_CheckError(result, "SspTns Merge", NULL);
    }

    for(i = 0; i < tsr->nnz-1; ++i) {
        // If two nnz has the same indices
        if(spt_SemiSparseTensorCompareIndices(tsr, i, tsr, i+1) == 0) {
            size_t col;
            for(col = 0; col < tsr->stride; ++col) {
                // Add them together
                tsr->values.values[(i+1)*tsr->stride + col] += tsr->values.values[i*tsr->stride + col];
            }
            sptAppendSizeVector(&collided, i);
        }
    }

    // Swap the last invalidated item with the last item
    i = collided.len;
    while(i) {
        --i;
        assert(tsr->nnz != 0);
        spt_SwapValues(tsr, collided.data[i], tsr->nnz-1, buffer);
        --tsr->nnz;
    }

    // Make sure all Vectors and Matrices have correct sizes
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            tsr->inds[i].len = tsr->nnz;
        }
    }
    tsr->values.nrows = tsr->nnz;

    sptFreeSizeVector(&collided);
    free(buffer);

    result = sptSemiSparseTensorSortIndex(tsr);
    spt_CheckError(result, "SspTns Merge", NULL);
    return 0;
}

static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            size_t eleind1 = tsr->inds[i].data[ind1];
            size_t eleind2 = tsr->inds[i].data[ind2];
            tsr->inds[i].data[ind1] = eleind2;
            tsr->inds[i].data[ind2] = eleind1;
        }
    }
    if(ind1 != ind2) {
        memcpy(buffer, &tsr->values.values[ind1*tsr->stride], tsr->stride * sizeof (sptScalar));
        memmove(&tsr->values.values[ind1*tsr->stride], &tsr->values.values[ind2*tsr->stride], tsr->stride * sizeof (sptScalar));
        memcpy(&tsr->values.values[ind2*tsr->stride], buffer, tsr->stride * sizeof (sptScalar));
    }
}

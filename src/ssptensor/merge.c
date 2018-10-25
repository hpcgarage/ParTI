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
#include <assert.h>
#include <string.h>

static void spt_SwapValues(sptSemiSparseTensor *tsr, sptNnzIndex ind1, sptNnzIndex ind2, sptValue buffer[]);

/**
 * Merge fibers with identical indices of an invalid semi sparse tensor, making it valid
 * @param tsr the semi sparse tensor to operate on
 */
int spt_SemiSparseTensorMergeValues(sptSemiSparseTensor *tsr) {
    int result;
    sptNnzIndex i;
    sptNnzIndexVector collided;
    sptValue *buffer;

    if(tsr->nnz == 0) {
        return 0;
    }

    buffer = malloc(tsr->stride * sizeof (sptValue));
    spt_CheckOSError(!buffer, "SspTns Merge");

    result = sptNewNnzIndexVector(&collided, 0, 0);
    if(result) {
        free(buffer);
        spt_CheckError(result, "SspTns Merge", NULL);
    }

    for(i = 0; i < tsr->nnz-1; ++i) {
        // If two nnz has the same indices
        if(spt_SemiSparseTensorCompareIndices(tsr, i, tsr, i+1) == 0) {
            sptIndex col;
            for(col = 0; col < tsr->stride; ++col) {
                // Add them together
                tsr->values.values[(i+1)*tsr->stride + col] += tsr->values.values[i*tsr->stride + col];
            }
            sptAppendNnzIndexVector(&collided, i);
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

    sptFreeNnzIndexVector(&collided);
    free(buffer);

    result = sptSemiSparseTensorSortIndex(tsr);
    spt_CheckError(result, "SspTns Merge", NULL);
    return 0;
}

static void spt_SwapValues(sptSemiSparseTensor *tsr, sptNnzIndex ind1, sptNnzIndex ind2, sptValue buffer[]) {
    sptIndex i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            sptIndex eleind1 = tsr->inds[i].data[ind1];
            sptIndex eleind2 = tsr->inds[i].data[ind2];
            tsr->inds[i].data[ind1] = eleind2;
            tsr->inds[i].data[ind2] = eleind1;
        }
    }
    if(ind1 != ind2) {
        memcpy(buffer, &tsr->values.values[ind1*tsr->stride], tsr->stride * sizeof (sptValue));
        memmove(&tsr->values.values[ind1*tsr->stride], &tsr->values.values[ind2*tsr->stride], tsr->stride * sizeof (sptValue));
        memcpy(&tsr->values.values[ind2*tsr->stride], buffer, tsr->stride * sizeof (sptValue));
    }
}

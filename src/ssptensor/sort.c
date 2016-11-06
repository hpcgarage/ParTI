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
#include <stdlib.h>
#include <string.h>

static void spt_QuickSortIndex(sptSemiSparseTensor *tsr, size_t l, size_t r, sptScalar buffer[]);
static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]);

/**
 * Reorder the elements in a semi sparse tensor lexicographically
 * @param tsr  the semi sparse tensor to operate on
 */
int sptSemiSparseTensorSortIndex(sptSemiSparseTensor *tsr) {
    sptScalar *buffer = malloc(tsr->stride * sizeof (sptScalar));
    spt_CheckOSError(!buffer, "SspTns SortIndex");
    spt_QuickSortIndex(tsr, 0, tsr->nnz, buffer);
    free(buffer);
    return 0;
}

static void spt_QuickSortIndex(sptSemiSparseTensor *tsr, size_t l, size_t r, sptScalar buffer[]) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SemiSparseTensorCompareIndices(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SemiSparseTensorCompareIndices(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j, buffer);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    spt_QuickSortIndex(tsr, l, i, buffer);
    spt_QuickSortIndex(tsr, i, r, buffer);
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
        memcpy(&tsr->values.values[ind1*tsr->stride], &tsr->values.values[ind2*tsr->stride], tsr->stride * sizeof (sptScalar));
        memcpy(&tsr->values.values[ind2*tsr->stride], buffer, tsr->stride * sizeof (sptScalar));
    }
}

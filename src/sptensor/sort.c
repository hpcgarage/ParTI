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
#include "sptensor.h"

static void spt_QuickSortIndex(sptSparseTensor *tsr, size_t l, size_t r);
static void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2);

/**
 * Reorder the elements in a sparse tensor lexicographically
 * @param tsr  the sparse tensor to operate on
 */
void sptSparseTensorSortIndex(sptSparseTensor *tsr) {
    spt_QuickSortIndex(tsr, 0, tsr->nnz);
    if(tsr->nmodes != 0) {
        tsr->sortkey = tsr->nmodes - 1;
    } else {
        tsr->sortkey = 0;
    }
}

static void spt_QuickSortIndex(sptSparseTensor *tsr, size_t l, size_t r) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareIndices(tsr, i, tsr, p) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareIndices(tsr, p, tsr, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        spt_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    spt_QuickSortIndex(tsr, l, i);
    spt_QuickSortIndex(tsr, i, r);
}

static void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2) {
    size_t i;
    sptScalar val1, val2;
    for(i = 0; i < tsr->nmodes; ++i) {
        size_t eleind1 = tsr->inds[i].data[ind1];
        size_t eleind2 = tsr->inds[i].data[ind2];
        tsr->inds[i].data[ind1] = eleind2;
        tsr->inds[i].data[ind2] = eleind1;
    }
    val1 = tsr->values.data[ind1];
    val2 = tsr->values.data[ind2];
    tsr->values.data[ind2] = val1;
    tsr->values.data[ind1] = val2;
}

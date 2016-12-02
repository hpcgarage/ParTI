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
#include "sptensor.h"
#include <assert.h>

static void spt_QuickSortAtMode(sptSparseTensor *tsr, size_t l, size_t r, size_t mode);
static int spt_SparseTensorCompareAtMode(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2, size_t mode);
static void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2);

/**
 * Reorder the elements in a sparse tensor lexicographically, but consider mode `mode` the last one
 * @param tsr  the sparse tensor to operate on
 * @param mode the mode to be considered the last
 */
void sptSparseTensorSortIndexAtMode(sptSparseTensor *tsr, size_t mode) {
    spt_QuickSortAtMode(tsr, 0, tsr->nnz, mode);
    tsr->sortkey = mode;
}

static void spt_QuickSortAtMode(sptSparseTensor *tsr, size_t l, size_t r, size_t mode) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(spt_SparseTensorCompareAtMode(tsr, i, tsr, p, mode) < 0) {
            ++i;
        }
        while(spt_SparseTensorCompareAtMode(tsr, p, tsr, j, mode) < 0) {
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
    spt_QuickSortAtMode(tsr, l, i, mode);
    spt_QuickSortAtMode(tsr, i, r, mode);
}

static int spt_SparseTensorCompareAtMode(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2, size_t mode) {
    size_t i;
    size_t eleind1, eleind2;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != mode) {
            eleind1 = tsr1->inds[i].data[ind1];
            eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    eleind1 = tsr1->inds[mode].data[ind1];
    eleind2 = tsr2->inds[mode].data[ind2];
    if(eleind1 < eleind2) {
        return -1;
    } else if(eleind1 > eleind2) {
        return 1;
    } else {
        return 0;
    }
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

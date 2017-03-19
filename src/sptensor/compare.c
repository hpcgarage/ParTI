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

/**
 * compare two indices from two identical or distinct sparse tensors lexicographically
 * @param tsr1 the first sparse tensor
 * @param loc1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param loc2 the order of the element in the second sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t loc1, const sptSparseTensor *tsr2, size_t loc2) {
    size_t i;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        size_t eleind1 = tsr1->inds[i].data[loc1];
        size_t eleind2 = tsr2->inds[i].data[loc2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}

/**
 * Comapre two index arrays lexicographically
 * @param inds1 the first indices to be compared
 * @param inds2 the second indices to be compared
 * @param len the length of both inds1 and inds2
 * @return 1 for inds1 < inds2; -1 for the other cases.
 */
int spt_SparseTensorCompareIndicesLT(size_t * const inds1, size_t * const inds2, size_t len) {

    size_t i;
    for(i = 0; i < len; ++i) {
        if(inds1[i] >= inds2[i]) {
            return -1;
        }
    }
    return 1;
}


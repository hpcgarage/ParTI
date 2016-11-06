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
#include <assert.h>

/**
 * compare two indices from two identical or distinct semi sparse tensors lexicographically
 * @param tsr1 the first semi sparse tensor
 * @param ind1 the order of the element in the first semi sparse tensor whose index is to be compared
 * @param tsr2 the second semi sparse tensor
 * @param ind2 the order of the element in the second semi sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
int spt_SemiSparseTensorCompareIndices(const sptSemiSparseTensor *tsr1, size_t ind1, const sptSemiSparseTensor *tsr2, size_t ind2) {
    size_t i;
    assert(tsr1->nmodes == tsr2->nmodes);
    assert(tsr1->mode == tsr2->mode);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != tsr1->mode) {
            size_t eleind1 = tsr1->inds[i].data[ind1];
            size_t eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    return 0;
}

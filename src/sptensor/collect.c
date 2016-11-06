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

/**
 * Collect zeros from a sparse tensor
 *
 * This function finds zero values from the sparse tensor,
 * move them to the end of the array, then truncate the array.
 * Orders are not kept, call sptSparseTensorSortIndex after this.
 */
void spt_SparseTensorCollectZeros(sptSparseTensor *tsr) {
    size_t i =  0;
    size_t nnz = tsr->nnz;
    size_t mode;
    while(i < nnz) {
        if(tsr->values.data[i] == 0) {
            for(mode = 0; mode < tsr->nmodes; ++mode) {
                tsr->inds[mode].data[i] = tsr->inds[mode].data[nnz-1];
            }
            tsr->values.data[i] = tsr->values.data[nnz-1];
            --nnz;
        } else {
            ++i;
        }
    }
    tsr->nnz = nnz;
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->inds[mode].len = nnz;
    }
    tsr->values.len = nnz;
}

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
#include <string.h>

static int spt_CompareIndices(const sptSemiSparseTensor *tsr, size_t el_idx, const size_t indices[]) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            if(tsr->inds[i].data[el_idx] < indices[i]) {
                return -1;
            } else if(tsr->inds[i].data[el_idx] > indices[i]) {
                return 1;
            }
        }
    }
    return 0;
}

/**
 * Insert or replace a single element into a semi sparse tensor
 * @param tsr     the tensor to operate on
 * @param indices the indices to locate the element to insert
 * @param value   the value of the element to insert
 */
int spt_SemiSparseTensorAppend(sptSemiSparseTensor *tsr, const size_t indices[], sptScalar value) {
    int result;
    int need_resize = 0;
    if(tsr->nnz == 0) {
        need_resize = 1;
    } else if(spt_CompareIndices(tsr, tsr->nnz-1, indices) != 0) {
        need_resize = 1;
    }
    if(need_resize) {
        size_t i;
        for(i = 0; i < tsr->nmodes; ++i) {
            if(i != tsr->mode) {
                result = sptAppendSizeVector(&tsr->inds[i], indices[i]);
                spt_CheckError(result, "SspTns Append", NULL);
            }
        }
        result = sptAppendMatrix(&tsr->values, NULL);
        spt_CheckError(result, "SspTns Append", NULL);
        memset(&tsr->values.values[tsr->nnz * tsr->stride], 0, tsr->nmodes * sizeof (sptScalar));
        ++tsr->nnz;
    }
    tsr->values.values[(tsr->nnz-1) * tsr->stride + indices[tsr->mode]] = value;
    return 0;
}

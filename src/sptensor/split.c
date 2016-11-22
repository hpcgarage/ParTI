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
 * Construct a sub-tensor from an existing tensor, using given constraints
 *
 * @param[out] out        the output tensor, uninitialized
 * @param[in]  tsr        the input tensor
 * @param[in]  limit_low  length `nmodes`, restrict the lower index bounds of each mode
 * @param[in]  limit_high length `nmodes`, restrict the upper index bounds of each mode
 *
 * Indices are compared using `limit_low <= index < limit_high`.
 * Free `out` after it is no longer needed.
 */
int spt_SplitSparseTensor(sptSparseTensor *dest, const sptSparseTensor *tsr, const size_t limit_low[], const size_t limit_high[]) {
    int result;
    size_t i, m;
    result = sptNewSparseTensor(dest, tsr->nmodes, tsr->ndims);
    spt_CheckError(result, "SpTns Split", NULL);
    
    for(i = 0; i < tsr->nnz; ++i) {
        int match = 1;
        for(m = 0; m < tsr->nmodes; ++m) {
            if(tsr->inds[m].data([i]) < limit_low[m] || tsr->inds[m].data([i]) >= limit_high[m]) {
                match = 0;
                break;
            }
        }
        if(match) {
            for(m = 0; m < tsr->nmodes; ++m) {
                result = sptAppendSizeVector(&dest->inds[m], tsr->inds[m].data[i]);
                spt_CheckError(result, "SpTns Split", NULL);
            }
            sptAppendVector(&dest->vals, tsr->vals.data[i]);
            spt_CheckError(result, "SpTns Split", NULL);
        }
    }
    
    return 0;
}

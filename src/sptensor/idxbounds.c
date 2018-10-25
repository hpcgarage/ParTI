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
#include <string.h>

/**
 * Calculate the lowest and highest index bounds of a given sparse tensor
 *
 * For any index `inds[i, m]` in sparse tensor `tsr`,
 * `inds_low[m] <= inds[i, m] < inds_high[m]`.
 *
 * @param[out] inds_low   low bound, length `tsr->nmodes`
 * @param[out] inds_high  high bound, length `tsr->nmodes`
 * @param[in]  tsr        a given sparse tensor
 */
void sptSparseTensorCalcIndexBounds(sptIndex inds_low[], sptIndex inds_high[], const sptSparseTensor *tsr) {

    if(tsr->nnz == 0) {
        memset(inds_low, 0, tsr->nmodes * sizeof inds_low[0]);
        memset(inds_high, 0, tsr->nmodes * sizeof inds_high[0]);
        return;
    }
    for(sptIndex m = 0; m < tsr->nmodes; ++m) {
        inds_low[m] = tsr->inds[m].data[0];
        inds_high[m] = tsr->inds[m].data[0] + 1;
        for(sptNnzIndex i = 1; i < tsr->nnz; ++i) {
            if(tsr->inds[m].data[i] < inds_low[m]) {
                inds_low[m] = tsr->inds[m].data[i];
            }
            if(tsr->inds[m].data[i] + 1 > inds_high[m]) {
                inds_high[m] = tsr->inds[m].data[i] + 1;
            }
        }
    }
}

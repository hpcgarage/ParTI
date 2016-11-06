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
 * Element wise multiply two sparse tensors
 * @param[out] Z the result of X*Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptSparseTensorDotMul(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {
    size_t i, j;
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "shape mismatch");
        }
    }

    sptNewSparseTensor(Z, X->nmodes, X->ndims);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    /* Multiply elements one by one, assume indices are ordered */
    i = 0;
    j = 0;
    while(i < X->nnz && j < Y->nnz) {
        int compare = spt_SparseTensorCompareIndices(X, i, Y, j);

        if(compare > 0) {
            ++j;
        } else if(compare < 0) {
            ++i;
        } else {
            for(size_t mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Z->inds[mode], Y->inds[mode].data[j]);
                spt_CheckError(result, "SpTns DotMul", NULL);
            }
            result = sptAppendVector(&Z->values, Y->values.data[j]);
            spt_CheckError(result, "SpTns DotMul", NULL);

            Y->values.data[Z->nnz] *= X->values.data[i];
            ++Z->nnz;
            ++i;
            ++j;
        }
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CPU  SpTns DotMul");
    sptFreeTimer(timer);

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z);
    return 0;
}

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

/**
 * Element wise subtract two sparse tensors
 * @param[out] Z the result of X-Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptSparseTensorSub(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {

    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns Sub", "shape mismatch");
    }
    for(sptIndex i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns Sub", "shape mismatch");
        }
    }

    sptNewSparseTensor(Z, X->nmodes, X->ndims);

    /* Add elements one by one, assume indices are ordered */
    sptNnzIndex i, j;
    int result;
    i = 0;
    j = 0;
    while(i < X->nnz && j < Y->nnz) {
        int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
        if(compare > 0) {
            for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendIndexVector(&Z->inds[mode], Y->inds[mode].data[j]);
                spt_CheckError(result, "SpTns Sub", NULL);
            }
            result = sptAppendValueVector(&Z->values, -Y->values.data[j]);
            spt_CheckError(result, "SpTns Sub", NULL);

            ++Z->nnz;
            ++j;
        } else if(compare < 0) {
            for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendIndexVector(&Z->inds[mode], X->inds[mode].data[i]);
                spt_CheckError(result, "SpTns Sub", NULL);
            }
            result = sptAppendValueVector(&Z->values, X->values.data[i]);
            spt_CheckError(result, "SpTns Sub", NULL);

            ++Z->nnz;
            ++i;
        } else {
            for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendIndexVector(&Z->inds[mode], X->inds[mode].data[i]);
                spt_CheckError(result, "SpTns Sub", NULL);
            }
            result = sptAppendValueVector(&Z->values, X->values.data[i] - Y->values.data[j]);
            spt_CheckError(result, "SpTns Sub", NULL);

            ++Z->nnz;
            ++i;
            ++j;
        }
    }
    /* Append remaining elements of X to Z */
    while(i < X->nnz) {
        for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
            result = sptAppendIndexVector(&Z->inds[mode], X->inds[mode].data[i]);
            spt_CheckError(result, "SpTns Sub", NULL);
        }
        result = sptAppendValueVector(&Z->values, X->values.data[i]);
        spt_CheckError(result, "SpTns Sub", NULL);
        ++Z->nnz;
        ++i;
    }
    /* Append remaining elements of Y to Z */
    while(i < X->nnz) {
        for(sptIndex mode = 0; mode < Y->nmodes; ++mode) {
            result = sptAppendIndexVector(&Z->inds[mode], Y->inds[mode].data[j]);
            spt_CheckError(result, "SpTns Sub", NULL);
        }
        result = sptAppendValueVector(&Z->values, -Y->values.data[j]);
        spt_CheckError(result, "SpTns Sub", NULL);
        ++Z->nnz;
        ++j;
    }
    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z, 1, 1);
    
    return 0;
}

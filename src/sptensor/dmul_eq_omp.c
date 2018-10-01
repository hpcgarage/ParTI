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
 * Openmp parallelized Element wise multiply two sparse tensors, with exactly the same nonzero
 * distribution.
 * @param[out] Z the result of X*Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptOmpSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y) {
    sptNnzIndex i;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(Y->nnz != X->nnz) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotMul", "nonzero distribution mismatch");
    }
    sptNnzIndex nnz = X->nnz;

    sptCopySparseTensor(Z, X, 1);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    #pragma omp parallel for
    for(i=0; i< nnz; ++i)
        Z->values.data[i] = X->values.data[i] * Y->values.data[i];

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CPU  SpTns DotMul");
    sptFreeTimer(timer);

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    sptSparseTensorSortIndex(Z, 1);
    
    return 0;
}

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
#include "../error/error.h"
#include <string.h>

/**
 * Convert sparse tensor to dense matrix
 *
 * @param dest pointer to an uninitialized matrix
 * @param src  pointer to a valid sparse tensor
 */
int sptSparseTensorToMatrix(sptMatrix *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    if(src->nmodes != 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns -> Mtx", "shape mismatch");
    }
    result = sptNewMatrix(dest, src->ndims[0], src->ndims[1]);
    spt_CheckError(result, "SpTns -> Mtx", NULL);
    memset(dest->values, 0, dest->nrows * dest->stride * sizeof (sptScalar));
    for(i = 0; i < src->nnz; ++i) {
        dest->values[src->inds[0].data[i] * dest->stride + src->inds[1].data[i]] = src->values.data[i];
    }
    return 0;
}

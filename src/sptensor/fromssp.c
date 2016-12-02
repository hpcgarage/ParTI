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
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Convert a semi sparse tensor into a sparse tensor
 * @param[out] dest    a pointer to an uninitialized sparse tensor
 * @param[in]  src     a pointer to a valid semi sparse tensor
 * @param      epsilon a small positive value, usually 1e-6, which is considered approximately equal to zero
 */
int sptSemiSparseTensorToSparseTensor(sptSparseTensor *dest, const sptSemiSparseTensor *src, sptScalar epsilon) {
    size_t i;
    int result;
    size_t nmodes = src->nmodes;
    assert(epsilon > 0);
    dest->nmodes = nmodes;
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SspTns -> SpTns");
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->nnz = 0;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SspTns -> SpTns");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&dest->inds[i], 0, src->nnz);
        spt_CheckError(result, "SspTns -> SpTns", NULL);
    }
    result = sptNewVector(&dest->values, 0, src->nnz);
    spt_CheckError(result, "SspTns -> SpTns", NULL);
    for(i = 0; i < src->nnz; ++i) {
        size_t j;
        for(j = 0; j < src->ndims[src->mode]; ++j) {
            sptScalar data = src->values.values[i*src->stride + j];
            int data_class = fpclassify(data);
            if(
                data_class == FP_NAN ||
                data_class == FP_INFINITE ||
                (data_class == FP_NORMAL && !(data < epsilon && data > -epsilon))
            ) {
                size_t m;
                for(m = 0; m < nmodes; ++m) {
                    if(m != src->mode) {
                        result = sptAppendSizeVector(&dest->inds[m], src->inds[m].data[i]);
                    } else {
                        result = sptAppendSizeVector(&dest->inds[src->mode], j);
                    }
                    spt_CheckError(result, "SspTns -> SpTns", NULL);
                }
                result = sptAppendVector(&dest->values, data);
                spt_CheckError(result, "SspTns -> SpTns", NULL);
                ++dest->nnz;
            }
        }
    }
    sptSparseTensorSortIndex(dest);
    return 0;
}

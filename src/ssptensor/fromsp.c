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
#include <stdlib.h>
#include <string.h>

/**
 * Convert a sparse tensor into a semi sparse tensor
 * @param[out] dest    a pointer to an uninitialized semi sparse tensor
 * @param[in]  src     a pointer to a valid sparse tensor
 * @param      mode    the mode to be stored in dense format
 */
int sptSparseTensorToSemiSparseTensor(sptSemiSparseTensor *dest, const sptSparseTensor *src, sptIndex mode) {
    sptIndex i;
    int result;
    sptIndex nmodes = src->nmodes;
    if(nmodes < 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns -> SspTns", "nmodes < 2");
    }
    dest->nmodes = nmodes;
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns -> SspTns");
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->mode = mode;
    dest->nnz = src->nnz;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SpTns -> SspTns");
    for(i = 0; i < nmodes; ++i) {
        if(i != mode) {
            result = sptCopyIndexVector(&dest->inds[i], &src->inds[i], 1);
        } else {
            result = sptNewIndexVector(&dest->inds[i], 0, 0);
        }
        spt_CheckError(result, "SpTns -> SspTns", NULL);
    }
    result = sptNewMatrix(&dest->values, dest->nnz, dest->ndims[mode]);
    spt_CheckError(result, "SpTns -> SspTns", NULL);
    dest->stride = dest->values.stride;
    memset(dest->values.values, 0, dest->nnz * dest->stride * sizeof (sptValue));
    for(i = 0; i < dest->nnz; ++i) {
        dest->values.values[i*dest->stride + src->inds[mode].data[i]] = src->values.data[i];
    }
    sptSemiSparseTensorSortIndex(dest);
    spt_SemiSparseTensorMergeValues(dest);
    
    return 0;
}

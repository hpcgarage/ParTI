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
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "ssptensor.h"

/**
 * Create a new semi sparse tensor
 * @param tsr    a pointer to an uninitialized semi sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param mode   the mode which will be stored in dense format
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, size_t nmodes, size_t mode, const size_t ndims[]) {
    size_t i;
    int result;
    if(nmodes < 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SspTns New", "nmodes < 2");
    }
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SspTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->mode = mode;
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SspTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&tsr->inds[i], 0, 0);
        spt_CheckError(result, "SspTns New", NULL);
    }
    tsr->stride = ((ndims[mode]-1)/8+1)*8;
    result = sptNewMatrix(&tsr->values, 0, tsr->stride);
    spt_CheckError(result, "SspTns New", NULL);
    return 0;
}

/**
 * Copy a semi sparse tensor
 * @param[out] dest a pointer to an uninitialized semi sparse tensor
 * @param[in]  src  a pointer to a valid semi sparse tensor
 */
int sptCopySemiSparseTensor(sptSemiSparseTensor *dest, const sptSemiSparseTensor *src) {
    size_t i;
    int result;
    assert(src->nmodes >= 2);
    dest->nmodes = src->nmodes;
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SspTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->mode = src->mode;
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SspTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        spt_CheckError(result, "SspTns Copy", NULL);
    }
    dest->stride = src->stride;
    result = sptCopyMatrix(&dest->values, &src->values);
    spt_CheckError(result, "SspTns Copy", NULL);
    return 0;
}

/**
 * Release any memory the semi sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeSizeVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeMatrix(&tsr->values);
}

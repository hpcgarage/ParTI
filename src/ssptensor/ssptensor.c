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
int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, sptIndex nmodes, sptIndex mode, const sptIndex ndims[]) {
    sptIndex i;
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
        result = sptNewIndexVector(&tsr->inds[i], 0, 0);
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
    sptIndex i;
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
        result = sptCopyIndexVector(&dest->inds[i], &src->inds[i], 1);
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
    sptIndex i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeMatrix(&tsr->values);
}



/**
 * Create a new semi sparse tensor
 * @param tsr    a pointer to an uninitialized semi sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param mode   the mode which will be stored in dense format
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSemiSparseTensorGeneral(sptSemiSparseTensorGeneral *tsr, sptIndex nmodes, const sptIndex ndims[], sptIndex ndmodes, const sptIndex dmodes[]) {
    sptIndex i;
    int result;
    if(nmodes < 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SspTns New", "nmodes < 2");
    }
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SspTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);

    tsr->ndmodes = ndmodes;
    tsr->dmodes = malloc(nmodes * sizeof *tsr->dmodes);
    spt_CheckOSError(!tsr->dmodes, "SspTns New");
    memcpy(tsr->dmodes, dmodes, nmodes * sizeof *tsr->dmodes);

    sptIndex nsmodes = nmodes - ndmodes;
    tsr->nnz = 0;
    tsr->inds = malloc(nsmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SspTns New");
    for(i = 0; i < nsmodes; ++i) {
        result = sptNewIndexVector(&tsr->inds[i], 0, 0);
        spt_CheckError(result, "SspTns New", NULL);
    }
    tsr->strides = malloc(ndmodes * sizeof *tsr->strides);
    for(i = 0; i < ndmodes; ++i) {
        tsr->strides[i] = ((ndims[dmodes[i]]-1)/8+1)*8;
    }
    result = sptNewMatrix(&tsr->values, 0, tsr->strides[0]);
    spt_CheckError(result, "SspTns New", NULL);
    return 0;
}

/**
 * Release any memory the semi sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSemiSparseTensorGeneral(sptSemiSparseTensorGeneral *tsr) {
    sptIndex i;
    for(i = 0; i < (tsr->nmodes - tsr->ndmodes); ++i) {
        sptFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->dmodes);
    free(tsr->strides);
    free(tsr->inds);
    sptFreeMatrix(&tsr->values);
}

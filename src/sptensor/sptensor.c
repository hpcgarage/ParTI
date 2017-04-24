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
#include <stdlib.h>
#include <string.h>

/**
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, size_t nmodes, const size_t ndims[]) {
    size_t i;
    int result;
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&tsr->inds[i], 0, 0);
        spt_CheckError(result, "SpTns New", NULL);
    }
    result = sptNewVector(&tsr->values, 0, 0);
    spt_CheckError(result, "SpTns New", NULL);
    return 0;
}

/**
 * Copy a sparse tensor
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    dest->nmodes = src->nmodes;
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SpTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        spt_CheckError(result, "SpTns Copy", NULL);
    }
    result = sptCopyVector(&dest->values, &src->values);
    spt_CheckError(result, "SpTns Copy", NULL);
    return 0;
}

/**
 * Release any memory the semi sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeSizeVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeVector(&tsr->values);
}


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
#include "hicoo.h"

/**
 * Create a new sparse tensor in HiCOO format
 * @param hitsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensorHiCOO(
    sptSparseTensorHiCOO *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nnz,
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits)
{
    sptIndex i;
    int result;

    hitsr->nmodes = nmodes;
    hitsr->sortorder = malloc(nmodes * sizeof hitsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        hitsr->sortorder[i] = i;
    }
    hitsr->ndims = malloc(nmodes * sizeof *hitsr->ndims);
    spt_CheckOSError(!hitsr->ndims, "HiSpTns New");
    memcpy(hitsr->ndims, ndims, nmodes * sizeof *hitsr->ndims);
    hitsr->nnz = nnz;

    /* Parameters */
    hitsr->sb_bits = sb_bits; // block size by nnz
    hitsr->sk_bits = sk_bits; // kernel size by nnz
    hitsr->sc_bits = sc_bits; // chunk size by blocks

    result = sptNewNnzIndexVector(&hitsr->kptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    result = sptNewNnzIndexVector(&hitsr->cptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);

    result = sptNewNnzIndexVector(&hitsr->bptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    hitsr->binds = malloc(nmodes * sizeof *hitsr->binds);
    spt_CheckOSError(!hitsr->binds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewBlockIndexVector(&hitsr->binds[i], 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }

    hitsr->einds = malloc(nmodes * sizeof *hitsr->einds);
    spt_CheckOSError(!hitsr->einds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewElementIndexVector(&hitsr->einds[i], 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }
    result = sptNewValueVector(&hitsr->values, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);


    return 0;
}

/**
 * Release any memory the HiCOO sparse tensor is holding
 * @param hitsr the tensor to release
 */
void sptFreeSparseTensorHiCOO(sptSparseTensorHiCOO *hitsr)
{
    sptIndex i;
    sptIndex nmodes = hitsr->nmodes;

    free(hitsr->sortorder);
    free(hitsr->ndims);

    sptFreeNnzIndexVector(&hitsr->kptr);
    sptFreeNnzIndexVector(&hitsr->cptr);

    sptFreeNnzIndexVector(&hitsr->bptr);
    for(i = 0; i < nmodes; ++i) {
        sptFreeBlockIndexVector(&hitsr->binds[i]);
        sptFreeElementIndexVector(&hitsr->einds[i]);
    }
    free(hitsr->binds);
    free(hitsr->einds);
    sptFreeValueVector(&hitsr->values);

    hitsr->nmodes = 0;
    hitsr->nnz = 0;
    hitsr->sb_bits = 0;
    hitsr->sk_bits = 0;
    hitsr->sc_bits = 0;
}
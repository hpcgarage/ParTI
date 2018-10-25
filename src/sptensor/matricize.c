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

int sptMatricize(sptSparseTensor const * const X,
    sptIndex const m,
    sptSparseMatrix * const A,
    int const transpose) {

    sptIndex const nmodes = X->nmodes;
    sptNnzIndex const nnz = X->nnz;
    sptIndex const * const ndims = X->ndims;
    sptValue const * const vals = X->values.data;

    /* Initialize sparse matrix A. */
    A->nnz = nnz;
    sptNewIndexVector(&(A->rowind), 0, nnz);
    sptNewIndexVector(&(A->colind), 0, nnz);
    sptNewValueVector(&(A->values), 0, nnz);

    /* Calculate the new_order of the rest modes except mode-m. */
    sptIndex * new_order = (sptIndex *)malloc((nmodes-1) * sizeof * new_order);
    for(sptIndex i=1; i<nmodes; ++i) {
        new_order[i-1] = (m+i) % nmodes;
    }
    /* Calculate strides for tensor X, from mode-(N-1) to mode-0. */
    sptNnzIndex * strides = (sptNnzIndex *)malloc((nmodes-1) * sizeof *strides);
    strides[nmodes-2] = 1;
    for(int i=nmodes-3; i>=0; --i) {
        sptIndex new_i = new_order[i];
        strides[i] = strides[i+1] * ndims[new_i+1];
    }


    if(transpose == 1) {    // mode-m as row
        A->nrows = ndims[m];
        A->ncols = 1;
        for(sptIndex i=0; i<nmodes-1; ++i) {
            sptIndex new_i = new_order[i];
            A->ncols *= ndims[new_i];
        }

        sptCopyIndexVector(&(A->rowind), &(X->inds[m]), 1);
        for(sptNnzIndex x=0; x<nnz; ++x) {
            sptNnzIndex col = 0;
            for(sptIndex i=0; i<nmodes-1; ++i) {
                sptIndex new_i = new_order[i];
                sptIndex index = X->inds[new_i].data[x];
                col += index * strides[i];
            }
            sptAppendIndexVector(&(A->colind), (sptIndex)col);  // maybe overflow
            sptAppendValueVector(&(A->values), vals[x]);
        }

    } else if(transpose == 0) {    // mode-m as column
        A->ncols = ndims[m];
        A->nrows = 1;
        for(sptIndex i=0; i<nmodes-1; ++i) {
            sptIndex new_i = new_order[i];
            A->nrows *= ndims[new_i];
        }

        sptCopyIndexVector(&(A->colind), &(X->inds[m]), 1);
        for(sptNnzIndex x=0; x<nnz; ++x) {
            sptNnzIndex row = 0;
            for(sptIndex i=0; i<nmodes-1; ++i) {
                sptIndex new_i = new_order[i];
                sptIndex index = X->inds[new_i].data[x];
                row += index * strides[i];
            }
            sptAppendIndexVector(&(A->colind), (sptIndex)row);
            sptAppendValueVector(&(A->values), vals[x]);
        }

    } else{
        spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Matricize", "incorrect transpose value");
    }

    free(strides);
    free(new_order);
    return 0;
}

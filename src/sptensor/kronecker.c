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
#include "sptensor.h"
#include <stdlib.h>
#include <string.h>

/**
 * Kronecker product of two sparse tensors
 * @param[out] Y the result of A(*)B, should be uninitialized
 * @param[in]  A the input A
 * @param[in]  B the input B
 */
int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B) {
    size_t nmodes;
    size_t mode;
    size_t *inds;
    size_t i, j;
    int result;
    if(A->nmodes != B->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns Kronecker", "shape mismatch");
    }
    nmodes = A->nmodes;
    inds = malloc(nmodes * sizeof *inds);
    spt_CheckOSError(!inds, "SpTns Kronecker");
    for(mode = 0; mode < nmodes; ++mode) {
        inds[mode] = A->ndims[mode] * B->ndims[mode];
    }
    result = sptNewSparseTensor(Y, nmodes, inds);
    free(inds);
    spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns Kronecker", "shape mismatch");
    /* For each element in A and B */
    for(i = 0; i < A->nnz; ++i) {
        for(j = 0; j < B->nnz; ++j) {
            /*
                Y[f(i1,j1), ..., f(i(N-1), j(N-1)] = a[i1, ..., i(N-1)] * b[j1, ..., j(N-1)]
                where f(in, jn) = jn + in * Jn
            */
            /* jli: (TODO). Append when acculumating a certain number (e.g. 10) of elements. 
                Don't do realloc only increasing length by one. 
                ! More important: The resulting Kronecker-product size is fixed, nnzA * nnzB. 
                Don't need realloc.

               sb: sptAppendSizeVector already do acculumating
            */
            for(mode = 0; mode < nmodes; ++mode) {
                sptAppendSizeVector(&Y->inds[mode], A->inds[mode].data[i] * B->ndims[mode] + B->inds[mode].data[j]);
            }
            sptAppendVector(&Y->values, A->values.data[i] * B->values.data[j]);
            ++Y->nnz;
        }
    }
    sptSparseTensorSortIndex(Y);
    return 0;
}

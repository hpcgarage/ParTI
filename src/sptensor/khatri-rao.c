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

/* jli: (TODO) Keep this function, but add another Khatri-Rao product for two dense matrices. */
/* jli: (Future TODO) Add Khatri-Rao product for two sparse matrices. */

int sptSparseTensorKhatriRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B) {
    size_t nmodes;
    size_t mode;
    size_t *inds;
    size_t i, j;
    int result;
    if(A->nmodes != B->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Khatri-Rao", "shape mismatch");
    }
    nmodes = A->nmodes;
    if(nmodes == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Khatri-Rao", "shape mismatch");
    }
    if(A->ndims[nmodes-1] != B->ndims[nmodes-1]) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Khatri-Rao", "shape mismatch");
    }
    inds = malloc(nmodes * sizeof *inds);
    spt_CheckOSError(!inds, "Khatri-Rao");
    for(mode = 0; mode < nmodes-1; ++mode) {
        inds[mode] = A->ndims[mode] * B->ndims[mode];
    }
    inds[nmodes-1] = A->ndims[mode];
    result = sptNewSparseTensor(Y, nmodes, inds);
    free(inds);
    spt_CheckError(result, "Khatri-Rao", NULL);
    /* For each element in A and B */
    for(i = 0; i < A->nnz; ++i) {
        for(j = 0; j < B->nnz; ++j) {
            if(A->inds[nmodes-1].data[i] == B->inds[nmodes-1].data[j]) {
                /*
                    Y[f(i0,j0), ..., f(i(N-2), j(N-2))] = a[i10 ..., i(N-2)] * b[j0, ..., j(N-2)]
                    where f(in, jn) = jn + in * Jn
                */
                for(mode = 0; mode < nmodes-1; ++mode) {
                    sptAppendSizeVector(&Y->inds[mode], A->inds[mode].data[i] * B->ndims[mode] + B->inds[mode].data[j]);
                }
                sptAppendSizeVector(&Y->inds[nmodes-1], A->inds[nmodes-1].data[i]);
                sptAppendVector(&Y->values, A->values.data[i] * B->values.data[j]);
                ++Y->nnz;
            }
        }
    }
    sptSparseTensorSortIndex(Y);
    return 0;
}

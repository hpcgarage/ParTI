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
#include <stdlib.h>
#include "matrix.h"
#include "mex.h"
#include "../sptmx.h"

spt_DefineSetScalar(spt_mxSetScalar, sptScalar)

void mexFunction1(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptMatrix:values", 1, "One", 1, "One");

    sptMatrix *mtx = spt_mxGetPointer(prhs[0], 0);

    mxDestroyArray(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(mtx->nrows, mtx->ncols, mxDOUBLE_CLASS, mxREAL);
    size_t i, j;
    for(i = 0; i < mtx->nrows; ++i) {
        for(j = 0; j < mtx->ncols; ++j) {
            size_t mxoffset = j * mtx->nrows + i;
            size_t sptoffset = i * mtx->stride + j;
            spt_mxSetScalar(plhs[0], mxoffset, mtx->values[sptoffset]);
        }
    }
}

void mexFunction2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptMatrix:values", 1, "One", 2, "Two");

    sptMatrix *mtx = spt_mxGetPointer(prhs[0], 0);
    size_t i = mxGetScalar(prhs[1])-1;

    mxDestroyArray(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    spt_mxSetScalar(plhs[0], 0, mtx->values[i]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs == 2) {
        mexFunction2(nlhs, plhs, nrhs, prhs);
    } else {
        mexFunction1(nlhs, plhs, nrhs, prhs);
    }
}

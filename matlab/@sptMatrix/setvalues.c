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

spt_DefineCastArray(spt_mxArrayToScalar, sptScalar)

void mexFunction2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptMatrix:setvalues", 0, "No", 2, "Two");

    mxArray *mxvalues = NULL;
    mexCallMATLAB(1, &mxvalues, 1, (mxArray **) &prhs[1], "transpose");

    sptMatrix *mtx = spt_mxGetPointer(prhs[0], 0);
    sptScalar *values = spt_mxArrayToScalar(mxvalues);
    size_t n = mxGetNumberOfElements(mxvalues);

    size_t i, j;
    for(i = 0; i < mtx->nrows; ++i) {
        for(j = 0; j < mtx->ncols; ++j) {
            size_t mxoffset = i * mtx->ncols + j;
            size_t sptoffset = i * mtx->stride + j;
            if(mxoffset < n) {
                mtx->values[mxoffset] = values[sptoffset];
            }
        }
    }

    mxDestroyArray(mxvalues);

    free(values);
}

void mexFunction3(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptMatrix:setvalues", 0, "No", 3, "Three");

    sptMatrix *mtx = spt_mxGetPointer(prhs[0], 0);
    size_t i = mxGetScalar(prhs[1])-1;
    sptScalar value = mxGetScalar(prhs[2]);

    mtx->values[i] = value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs == 3) {
        mexFunction3(nlhs, plhs, nrhs, prhs);
    } else {
        mexFunction2(nlhs, plhs, nrhs, prhs);
    }
}

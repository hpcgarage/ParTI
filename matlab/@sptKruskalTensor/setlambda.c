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
#include <stdlib.h>
#include "matrix.h"
#include "mex.h"
#include "../sptmx.h"

spt_DefineCastArray(spt_mxArrayToScalar, sptScalar)

void mexFunction2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptKruskalTensor:setlambda", 0, "No", 2, "Two");

    sptKruskalTensor *ktsr = spt_mxGetPointer(prhs[0], 0);
    sptScalar *values = spt_mxArrayToScalar(prhs[1]);
    size_t n = mxGetNumberOfElements(prhs[1]);

    size_t i;
    for(i = 0; i < ktsr->rank&& i < n; ++i) {
        ktsr->lambda[i] = values[i];
    }

    free(values);
}

void mexFunction3(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptKruskalTensor:setlambda", 0, "No", 3, "Three");

    sptKruskalTensor *ktsr = spt_mxGetPointer(prhs[0], 0);
    size_t i = mxGetScalar(prhs[1])-1;
    sptScalar value = mxGetScalar(prhs[2]);

    ktsr->lambda[i] = value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs == 3) {
        mexFunction3(nlhs, plhs, nrhs, prhs);
    } else {
        mexFunction2(nlhs, plhs, nrhs, prhs);
    }
}

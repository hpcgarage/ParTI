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
    spt_mxCheckArgs("sptSparseTensor:values", 1, "One", 1, "One");

    sptSparseTensor *tsr = spt_mxGetPointer(prhs[0], 0);

    mexCallMATLAB(1, plhs, 0, NULL, "sptSizeVector");
    mxArray *args[] = {plhs[0], 1, tsr->nmodes};
    mexCallMATLAB(1, plhs, 3, args, "repmat");

    size_t m;
    for(m = 0; m < tsr->nmodes; ++m) {
        spt_mxSetPointer(plhs[0], m, &tsr->inds[m]);
    }
}

void mexFunction2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptSparseTensor:values", 1, "One", 2, "Two");

    sptSparseTensor *tsr = spt_mxGetPointer(prhs[0], 0);
    size_t m = mxGetScalar(prhs[1])-1;

    mexCallMATLAB(1, plhs, 0, NULL, "sptSizeVector");
    spt_mxSetPointer(plhs[0], 0, &tsr->inds[m]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs == 2) {
        mexFunction2(nlhs, plhs, nrhs, prhs);
    } else {
        mexFunction1(nlhs, plhs, nrhs, prhs);
    }
}

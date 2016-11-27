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
#include "sptmx.h"

spt_DefineCastArray(spt_mxArrayToSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptNewSparseTensor", 1, "One", 2, "Two");

    size_t nmodes = mxGetScalar(prhs[0]);
    if(mxGetNumberOfElements(prhs[1]) != nmodes) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "length of ndims should be nmodes");
    }
    size_t *ndims = spt_mxArrayToSize(prhs[1]);

    sptSparseTensor *tsr = malloc(sizeof *tsr);
    int result = sptNewSparseTensor(tsr, nmodes, ndims);
    free(ndims);
    if(result) {
        free(tsr);
        tsr = NULL;
    }

    mexCallMATLAB(nlhs, plhs, 0, NULL, "sptSparseTensor");
    spt_mxSetPointer(plhs[0], tsr);
}

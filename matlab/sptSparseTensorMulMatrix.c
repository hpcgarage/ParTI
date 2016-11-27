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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptSparseTensorMulMatrix", 1, "One", 3, "Three");

    sptSparseTensor *X = spt_mxGetPointer(prhs[0], 0);
    sptMatrix *U = spt_mxGetPointer(prhs[1], 0);
    size_t mode = mxGetScalar(prhs[2]);

    sptSemiSparseTensor *Y = malloc(sizeof *Y);
    int result = sptSparseTensorMulMatrix(Y, X, U, mode);
    if(result) {
        free(Y);
        Y = NULL;
    }

    mexCallMATLAB(nlhs, plhs, 0, NULL, "sptSemiSparseTensor");
    spt_mxSetPointer(plhs[0], 0, Y);
}

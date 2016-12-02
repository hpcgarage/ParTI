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
#include "sptmx.h"

spt_DefineCastArray(spt_mxArrayToSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptNewMatrix", 1, "One", 2, "Two");

    size_t nrows = mxGetScalar(prhs[0]);
    size_t ncols = mxGetScalar(prhs[1]);

    sptMatrix *mtx = malloc(sizeof *mtx);
    int result = sptNewMatrix(mtx, nrows, ncols);
    if(result) {
        free(mtx);
        mtx = NULL;
    }

    mexCallMATLAB(nlhs, plhs, 0, NULL, "sptMatrix");
    spt_mxSetPointer(plhs[0], 0, mtx);
}

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

spt_DefineSetScalar(spt_mxSetSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptSparseTensor:sortkey", 1, "One", 1, "One");

    sptSparseTensor *tsr = spt_mxGetPointer(prhs[0], 0);

    mxDestroyArray(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    spt_mxSetSize(plhs[0], 0, tsr->sortkey);
}

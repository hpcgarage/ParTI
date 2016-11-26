#include <SpTOL.h>
#include <stdlib.h>
#include "matrix.h"
#include "mex.h"
#include "sptmx.h"

spt_DefineCastArray(spt_mxArrayToSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 1) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "One input required.");
    }
    if(nlhs != 0) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "No outputs required.");
    }

    sptSparseTensor *tsr = spt_mxGetPointer(prhs[0]);
    sptFreeSparseTensor(tsr);
    free(tsr);
}

#include <SpTOL.h>
#include <stdlib.h>
#include "matrix.h"
#include "mex.h"
#include "sptmx.h"

spt_DefineSetScalar(spt_mxSetSize, size_t)
spt_DefineCastArray(spt_mxArrayToSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 2) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "Two inputs required.");
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "One output required.");
    }
    if(!mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "ndims should be double");
    }

    mexCallMATLAB(nlhs, plhs, 0, NULL, "sptSparseTensor");
    size_t nmodes = mxGetScalar(prhs[0]);
    if(mxGetNumberOfElements(prhs[1]) != nmodes) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "length of ndims should be nmodes");
    }
    size_t *ndims = spt_mxArrayToSize(prhs[1]);

    sptSparseTensor *tsr = malloc(sizeof *tsr);
    int result = sptNewSparseTensor(tsr, nmodes, ndims);
    if(result) {
        free(tsr);
        tsr = NULL;
    }

    mxArray *mxptr = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    spt_mxSetSize(mxptr, 0, (size_t) tsr);
    mxSetProperty(plhs[0], 0, "ptr", mxptr);
    mxDestroyArray(mxptr);

    free(ndims);
}

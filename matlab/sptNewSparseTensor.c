#include <SpTOL.h>
#include <stdlib.h>
#include "matrix.h"
#include "sptCast.h"

spt_DefineCastArray(spt_mxArrayToSize, size_t)
spt_DefineSetScalar(spt_mxSetScalar, int)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 3) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "Three inputs required.");
    }
    if(!mxIsDouble(prhs[2])) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "ndims should be double");
    }

    sptSparseTensor **tsr = mxGetData(mxGetProperty(prhs[0], 0, "ptr"));
    size_t nmodes = mxGetScalar(prhs[1]);
    if(mxGetNumberOfElements(prhs[2]) != nmodes) {
        mexErrMsgIdAndTxt("SpTOL:sptNewSparseTensor", "length of ndims should be nmodes");
    }
    size_t *ndims = spt_mxArrayToSize(prhs[2]);

    *tsr = malloc(sizeof **tsr);
    int result = sptNewSparseTensor(tsr, nmodes, ndims);

    free(ndims);
    if(nlhs >= 1) {
        spt_mxSetScalar(plhs[0], 0, result);
    }
}

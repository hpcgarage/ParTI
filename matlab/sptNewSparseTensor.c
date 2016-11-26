#include <SpTOL.h>
#include <stdlib.h>
#include "matrix.h"

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
    const double *ndims_double = mxGetPr(prhs[2]);
    size_t *ndims = malloc(nmodes * sizeof (size_t));
    size_t i;
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = ndims_double;
    }
    *tsr = malloc(sizeof **tsr);
    int result = sptNewSparseTensor(tsr, nmodes, ndims);
}

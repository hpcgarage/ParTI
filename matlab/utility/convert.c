#include "matrix.h"

double *sptmx_ArrayToDouble(const mxArray *arr) {
    mxClassID clsid = mxGetClassID(arr);
    double *result = malloc(mxGetNumberOfElements * sizeof (double));
    switch(clsid) {
    }
}

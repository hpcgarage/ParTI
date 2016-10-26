#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorDivScalar(sptSparseTensor *X, sptScalar a) {
    if(a != 0) {
        size_t i;
        #pragma omp parallel for schedule(static)
        for(i = 0; i < X->nnz; ++i) {
            X->values.data[i] /= a;
        }
        return 0;
    } else {
        spt_CheckError(SPTERR_ZERO_DIVISION, "SpTns Div", "divide by zero");
    }
}

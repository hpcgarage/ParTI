#include <SpTOL.h>

int sptSparseTensorMulScalar(sptSparseTensor *X, sptScalar a) {
    if(a != 0) {
        size_t i;
        for(i = 0; i < X->nnz; ++i) {
            X->values[i] *= a;
        }
    } else {
        X->nnz = 0;
    }
    return 0;
}

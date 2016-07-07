#include <SpTOL.h>

int sptSparseTensorDivScalar(sptSparseTensor *X, sptScalar a) {
    size_t i;
    for(i = 0; i < X->nnz; ++i) {
        X->values.data[i] /= a;
    }
    return 0;
}

#include <SpTOL.h>

int sptSparseTensorMulScalar(sptSparseTensor *X, sptScalar a) {
    if(a != 0) {
        size_t i;
        #pragma omp parallel for schedule(static)
        for(i = 0; i < X->nnz; ++i) {
            X->values.data[i] *= a;
        }
    } else {
        X->nnz = 0;
        X->values.len = 0;
    }
    return 0;
}

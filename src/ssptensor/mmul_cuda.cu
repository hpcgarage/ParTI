#include <SpTOL.h>


int sptCudaSparseTensorMulMatrixAsSemiSparseTensor(
    sptSemiSparseTensor *Y,
    const sptSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
    cudaSetDevice(0);
    return -1;
}

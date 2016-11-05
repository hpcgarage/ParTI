#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fY;
    sptSparseTensor X, spY;
    sptSemiSparseTensor Y;
    sptMatrix U;
    size_t mode = 0;
    size_t R = 16;
    int cuda_dev_id = -2;
    int niters = 5;

    if(argc < 3) {
        printf("Usage: %s X mode [cuda_dev_id, R, Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    assert(fX != NULL);
    assert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    sscanf(argv[2], "%zu", &mode);
    if(argc >= 4) {
        sscanf(argv[3], "%d", &cuda_dev_id);
        sptCudaSetDevice(cuda_dev_id);
    }
    if(argc >= 5) {
        sscanf(argv[4], "%zu", &R);
    }

    printf("Tensor: %s, TTM mode %zu\n", argv[1], mode);

    assert(sptRandomizeMatrix(&U, X.ndims[mode], R) == 0);

    /* We have niters+1 iterations, the first is warm-up */
    assert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    for(int it=0; it<niters; ++it) {
        sptFreeSemiSparseTensor(&Y);
        assert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }
    for(int it=0; it<niters+1; ++it) {
        sptFreeSemiSparseTensor(&Y);
        assert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }
    if(cuda_dev_id >= 0) {
        cudaDeviceReset();
        unsetenv("SPTOL_TTM_KERNEL");
        for(int it=0; it<niters+1; ++it) {
            sptFreeSemiSparseTensor(&Y);
            assert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        }
        cudaDeviceReset();
        setenv("SPTOL_TTM_KERNEL", "naive", 1);
        for(int it=0; it<niters+1; ++it) {
            sptFreeSemiSparseTensor(&Y);
            assert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        }
    }

    if(argc >= 6) {
        assert(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);
    }

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    if(argc >= 6) {
        fY = fopen(argv[5], "w");
        assert(fY != NULL);
        assert(sptDumpSparseTensor(&spY, 1, fY) == 0);
        fclose(fY);
        sptFreeSparseTensor(&spY);
    }

    return 0;
}

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fU, *fY;
    sptSparseTensor X, spU, spY;
    sptSemiSparseTensor Y;
    sptMatrix U;
    size_t mode = 0;
    int cuda_dev_id = -2;

    if(argc < 5) {
        printf("Usage: %s X U Y mode [cuda_dev_id]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    assert(fX != NULL);
    assert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    fU = fopen(argv[2], "r");
    assert(fU != NULL);
    assert(sptLoadSparseTensor(&spU, 1, fU) == 0);
    fclose(fU);

    sscanf(argv[4], "%zu", &mode);
    if(argc >= 6) {
        sscanf(argv[5], "%d", &cuda_dev_id);
    }

    assert(sptSparseTensorToMatrix(&U, &spU) == 0);
    sptFreeSparseTensor(&spU);

    sptTimer timer;
    sptNewTimer(&timer, cuda_dev_id >= 0);
    sptStartTimer(timer);

    if(cuda_dev_id == -2) {
        assert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(cuda_dev_id == -1) {
        assert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        assert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }

    sptStopTimer(timer);
    printf("Elapsed %.09lf s\n", sptElapsedTime(timer));
    sptFreeTimer(timer);

    assert(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    fY = fopen(argv[3], "w");
    assert(fY != NULL);
    assert(sptDumpSparseTensor(&spY, 1, fY) == 0);
    fclose(fY);

    sptFreeSparseTensor(&spY);

    return 0;
}

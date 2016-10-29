#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char *argv[]) {
    FILE *fX, *fY, *fo;
    sptSparseTensor X, Y, Z;
    int cuda_dev_id = -2;
    int niters = 5;

    if(argc < 3) {
        printf("Usage: %s X Y [cuda_dev_id, niters, out]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    assert(fX != NULL);
    assert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    fY = fopen(argv[1], "r");
    assert(fY != NULL);
    assert(sptLoadSparseTensor(&Y, 1, fY) == 0);
    fclose(fY);

    if(argc >= 4) {
        sscanf(argv[3], "%d", &cuda_dev_id);
    }
    if(argc >= 5) {
        sscanf(argv[4], "%d", &niters);
    }
    assert(niters >= 1);

    // sptSparseTensorSortIndex(&a);
    // sptSparseTensorSortIndex(&b);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        assert(sptSparseTensorDotMulEq(&Z, &X, &Y) == 0);
    }  else if(cuda_dev_id >= 0) {
        sptCudaSetDevice(cuda_dev_id);
        assert(sptCudaSparseTensorDotMul(&Z, &X, &Y) == 0);
    }

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            assert(sptSparseTensorDotMulEq(&Z, &X, &Y) == 0);
        }  else if(cuda_dev_id >= 0) {
            sptCudaSetDevice(cuda_dev_id);
            assert(sptCudaSparseTensorDotMul(&Z, &X, &Y) == 0);
        }
    }

    if(argc >= 6) {
        fo = fopen(argv[5], "w");
        assert(fo != NULL);
        assert(sptDumpSparseTensor(&Z, 1, fo) == 0);
        fclose(fo);
    }

    sptFreeSparseTensor(&X);
    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&Z);

    return 0;
}

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fo;
    sptSparseTensor X;
    sptMatrix ** U;
    sptSizeVector mats_order;
    sptVector scratch;
    size_t mode = 0;
    size_t R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;

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
    }
    if(argc >= 5) {
        sscanf(argv[4], "%zu", &R);
    }

    size_t nmodes = X.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(size_t m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    size_t max_ndims = 0;
    for(size_t m=0; m<nmodes; ++m) {
      // assert(sptRandomizeMatrix(U[m], X.ndims[m], R) == 0);
      assert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
      assert(sptConstantMatrix(U[m], 1) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    assert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    assert(sptConstantMatrix(U[nmodes], 0) == 0);
    size_t stride = U[0]->stride;


    sptNewSizeVector(&mats_order, nmodes-1, nmodes-1);
    size_t j = 0;
    for(int m=nmodes-1; m>=0; --m) {
        if(m != mode) {
            mats_order.data[j] = m;
            ++ j;
        }
    }


    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        sptNewVector(&scratch, R, R);
        sptConstantVector(&scratch, 0);
        assert(sptMTTKRP(&X, U, &mats_order, mode, &scratch) == 0);
        sptFreeVector(&scratch);
    } else if(cuda_dev_id == -1) {
        #pragma omp parallel 
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        sptNewVector(&scratch, X.nnz * stride, X.nnz * stride);
        sptConstantVector(&scratch, 0);
        assert(sptOmpMTTKRP(&X, U, &mats_order, mode, &scratch) == 0);
        sptFreeVector(&scratch);
    } else {
       sptCudaSetDevice(cuda_dev_id);
       assert(sptCudaMTTKRP(&X, U, &mats_order, mode, &scratch) == 0);
    }
    // sptDumpMatrix(U[nmodes], stdout);
    

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            nthreads = 1;
            sptNewVector(&scratch, R, R);
            sptConstantVector(&scratch, 0);
            assert(sptMTTKRP(&X, U, &mats_order, mode, &scratch) == 0);
            sptFreeVector(&scratch);
        } else if(cuda_dev_id == -1) {
            #pragma omp parallel 
            {
                nthreads = omp_get_num_threads();
            }
            printf("nthreads: %d\n", nthreads);
            sptNewVector(&scratch, X.nnz * stride, X.nnz * stride);
            sptConstantVector(&scratch, 0);
            assert(sptOmpMTTKRP(&X, U, &mats_order, mode, &scratch) == 0);
            sptFreeVector(&scratch);
        } else {
           sptCudaSetDevice(cuda_dev_id);
           assert(sptCudaMTTKRP(&X, U, &mats_order, mode, &scratch) == 0);
        }
    }


    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&X);
    sptFreeSizeVector(&mats_order);

    if(argc >= 6) {
        fo = fopen(argv[5], "w");
        assert(fo != NULL);
        assert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);

    return 0;
}

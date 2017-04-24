/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"

int main(int argc, char const *argv[]) {
    FILE *fX, *fo;
    sptSparseTensor X;
    sptMatrix ** U;
    sptVector scratch;
    size_t mode = 0;
    size_t R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;

    if(argc < 4) {
        printf("Usage: %s X mode impl_num [cuda_dev_id, R, Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    printf("input file: %s\n", argv[1]); fflush(stdout);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    printf("Tensor ndims:\n");
    spt_DumpArray(X.ndims, X.nmodes, 0, stdout);
    printf("Tensor NNZ: %zu\n", X.nnz);

    sscanf(argv[2], "%zu", &mode);
    sscanf(argv[3], "%d", &impl_num);
    if(argc >= 5) {
        sscanf(argv[4], "%d", &cuda_dev_id);
    }
    if(argc >= 6) {
        sscanf(argv[5], "%zu", &R);
    }

    size_t nmodes = X.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(size_t m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    size_t max_ndims = 0;
    for(size_t m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    size_t stride = U[0]->stride;


    size_t * mats_order = (size_t*)malloc(nmodes * sizeof(size_t));
    mats_order[0] = mode;
    for(size_t i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;
    printf("mats_order:\n");
    spt_DumpArray(mats_order, nmodes, 0, stdout);


    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        sptNewVector(&scratch, R, R);
        sptConstantVector(&scratch, 0);
        sptAssert(sptMTTKRP(&X, U, mats_order, mode, &scratch) == 0);
        sptFreeVector(&scratch);
    } else if(cuda_dev_id == -1) {
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        sptNewVector(&scratch, X.nnz * stride, X.nnz * stride);
        sptConstantVector(&scratch, 0);
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, &scratch) == 0);
        sptFreeVector(&scratch);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        // sptAssert(sptCudaMTTKRP(&X, U, mats_order, mode, impl_num) == 0);
        sptAssert(sptCudaMTTKRPOneKernel(&X, U, mats_order, mode, impl_num) == 0);
    }

    for(int it=0; it<niters; ++it) {
        sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
        if(cuda_dev_id == -2) {
            nthreads = 1;
            sptNewVector(&scratch, R, R);
            sptConstantVector(&scratch, 0);
            sptAssert(sptMTTKRP(&X, U, mats_order, mode, &scratch) == 0);
            sptFreeVector(&scratch);
        } else if(cuda_dev_id == -1) {
            #pragma omp parallel
            {
                nthreads = omp_get_num_threads();
            }
            printf("nthreads: %d\n", nthreads);
            sptNewVector(&scratch, X.nnz * stride, X.nnz * stride);
            sptConstantVector(&scratch, 0);
            sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, &scratch) == 0);
            sptFreeVector(&scratch);
        } else {
            sptCudaSetDevice(cuda_dev_id);
            // sptAssert(sptCudaMTTKRP(&X, U, mats_order, mode, impl_num) == 0);
            sptAssert(sptCudaMTTKRPOneKernel(&X, U, mats_order, mode, impl_num) == 0);
        }
    }



    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&X);
    free(mats_order);

    if(argc >= 7) {
        fo = fopen(argv[6], "w");
        sptAssert(fo != NULL);
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);

    return 0;
}

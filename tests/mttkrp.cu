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
#include <getopt.h>
#include <omp.h>
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"

int main(int argc, char const *argv[]) {
    FILE *fX = NULL, *fo = NULL;
    sptSparseTensor X;
    sptMatrix ** U;
    sptMatrix ** copy_U;
    size_t mode = 0;
    size_t R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int use_reduce = 1; // todo: determine two omp parallel cases
    int nt = 1;
    printf("niters: %d\n", niters);

    for(;;) {
        static struct option long_options[] = {
            {"mode", required_argument, 0, 'm'},
            {"impl-num", required_argument, 0, 'i'},
            {"cuda-dev-id", required_argument, 0, 'd'},
            {"nt", optional_argument, 0, 't'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c;
        c = getopt_long(argc, const_cast<char *const *>(argv), "m:i:d:r:y:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'm':
            sscanf(optarg, "%zu", &mode);
            break;
        case 'i':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%zu", &R);
            break;
        case 'y':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        default:
            abort();
        }
    }

    if(optind >= argc) {
        printf("Usage: %s [options] X\n\n", argv[0]);
        printf("Options: -m MODE, --mode=MODE\n");
        printf("         -i IMPL_NUM, --impl-num=IMPL_NUM\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -r R\n");
        printf("         -y Y\n");
        printf("         -t NTHREADS, --nt=NT\n");
        printf("\n");
        return 1;
    }

    fX = fopen(argv[optind], "r");
    sptAssert(fX != NULL);
    printf("input file: %s\n", argv[optind]); fflush(stdout);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    sptSparseTensorStatus(&X, stdout);

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

    /* Set zeros for temporary copy_U, for mode-"mode" */
    char * bytestr;
    if(cuda_dev_id == -1 && use_reduce == 1) {
        copy_U = (sptMatrix **)malloc(nt * sizeof(sptMatrix*));
        for(int t=0; t<nt; ++t) {
            copy_U[t] = (sptMatrix *)malloc(sizeof(sptMatrix));
            sptAssert(sptNewMatrix(copy_U[t], X.ndims[mode], R) == 0);
            sptAssert(sptConstantMatrix(copy_U[t], 0) == 0);
        }
        size_t bytes = nt * X.ndims[mode] * R * sizeof(sptScalar);
        bytestr = sptBytesString(bytes);
        printf("MODE MATRIX COPY=%s\n\n", bytestr);
    }

    size_t * mats_order = (size_t*)malloc(nmodes * sizeof(size_t));
    mats_order[0] = mode;
    for(size_t i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;
    // printf("mats_order:\n");
    // spt_DumpArray(mats_order, nmodes, 0, stdout);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    } else if(cuda_dev_id == -1) {
        printf("nt: %d\n", nt);
        if(use_reduce == 1) {
            printf("sptOmpMTTKRP_Reduce:\n");
            sptAssert(sptOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
        } else {
            printf("sptOmpMTTKRP:\n");
            sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
        }
    } else {
        sptCudaSetDevice(cuda_dev_id);
        // sptAssert(sptCudaMTTKRP(&X, U, mats_order, mode, impl_num) == 0);
        sptAssert(sptCudaMTTKRPOneKernel(&X, U, mats_order, mode, impl_num) == 0);
    }

    
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            nthreads = 1;
            sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            if(use_reduce == 1) {
                sptAssert(sptOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
            } else {
                sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
            }
        } else {
            #if 0
           switch(ncudas) {
           case 1:
             sptCudaSetDevice(cuda_dev_id);
             sptAssert(sptCudaMTTKRP(&X, U, &mats_order, mode) == 0);
             break;
           case 2:
             sptCudaSetDevice(cuda_dev_id);
             sptCudaSetDevice(cuda_dev_id+1);
             printf("====\n");
             sptAssert(sptCudaMTTKRP(csX, U, &mats_order, mode) == 0);
             sptAssert(sptCudaMTTKRP(csX+1, U, &mats_order, mode) == 0);
             break;
           }
           #endif
            sptCudaSetDevice(cuda_dev_id);
            // sptAssert(sptCudaMTTKRP(&X, U, mats_order, mode, impl_num) == 0);
            sptAssert(sptCudaMTTKRPOneKernel(&X, U, mats_order, mode, impl_num) == 0);
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
    sptFreeTimer(timer);


    if(cuda_dev_id == -1 && use_reduce == 1) {
        for(int t=0; t<nt; ++t) {
            sptFreeMatrix(copy_U[t]);
        }
        free(copy_U);
        free(bytestr);
    }
    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&X);
    free(mats_order);

    if(fo != NULL) {
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);

    return 0;
}

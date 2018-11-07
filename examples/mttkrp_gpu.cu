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

void print_usage(int argc, char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 or 1 or 2 for third-order tensors. Default:0.)\n");
    printf("         -s sortcase, --sortcase=SORTCASE (0:default,1,2,3,4. Different tensor sorting.)\n");
    printf("         -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (only for sortcase=3)\n");
    printf("         -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (only for sortcase=3)\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=CUDA_DEV_ID (>=0:GPU device id)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         GPU options: \n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM (11, 12, 15, where 15 should be the best case)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;
    sptMatrix ** U;

    sptIndex mode = 0;
    sptIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int use_reduce = 1; // Need to choose from two omp parallel approaches
    int nt = 1;
    /* sortcase:
     * = 0 : the same with the old COO code.
     * = 1 : best case. Sort order: [mode, (ordered by increasing dimension sizes)]
     * = 2 : worse case. Sort order: [(ordered by decreasing dimension sizes)]
     * = 3 : Z-Morton ordering (same with HiCOO format order)
     * = 4 : random shuffling.
     */
    int sortcase = 0;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    printf("niters: %d\n", niters);

    if(argc <= 3) { // #Required arguments
        print_usage(argc, argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"mode", required_argument, 0, 'm'},
            {"output", optional_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"sortcase", optional_argument, 0, 's'},
            {"impl-num", optional_argument, 0, 'p'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:k:s:p:d:r:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            sptAssert(fi != NULL);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 's':
            sscanf(optarg, "%d", &sortcase);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u"PARTI_SCN_INDEX, &R);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argc, argv);
            exit(1);
        }
    }

    printf("mode: %"PARTI_PRI_INDEX "\n", mode);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    printf("sortcase: %d\n", sortcase);
    sptAssert(cuda_dev_id >= 0);

    /* Load a sparse tensor from file as it is */
    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    /* Sort sparse tensor */
    sptIndex * mode_order = (sptIndex*) malloc(X.nmodes * sizeof(*mode_order));
    memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
    switch (sortcase) {
        case 0:
            sptSparseTensorSortIndex(&X, 1);
            break;
        case 1:
            sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
            sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
            break;
        case 2:
            sptGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
            sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
            break;
        case 3:
            /* Pre-process tensor, the same with the one used in HiCOO.
             * Only difference is not setting kptr and kschr in this function.
             */
            sptSparseTensorMixedOrder(&X, sb_bits, sk_bits, 1);
            break;
        case 4:
            // sptGetBestModeOrder(mode_order, 0, X.ndims, X.nmodes);
            sptGetRandomShuffleElements(&X);
            break;
        default:
            printf("Wrong sortcase number, reset by -s. \n");
    }
    if(sortcase != 0) {
        printf("mode_order:\n");
        sptDumpIndexArray(mode_order, X.nmodes, stdout);
    }

    sptSparseTensorStatus(&X, stdout);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    sptIndex nmodes = X.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    sptIndex stride = U[0]->stride;

    sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    switch (sortcase) {
    case 0:
    case 3:
    case 4:
        mats_order[0] = mode;
        for(sptIndex i=1; i<nmodes; ++i)
            mats_order[i] = (mode+i) % nmodes;
        break;
    case 1: // Reverse of mode_order except the 1st one
        mats_order[0] = mode;
        for(sptIndex i=1; i<nmodes; ++i)
            mats_order[i] = mode_order[nmodes - i];
        break;
    case 2: // Totally reverse of mode_order
        for(sptIndex i=0; i<nmodes; ++i)
            mats_order[i] = mode_order[nmodes - i];
        break;
    }

    /* For warm-up caches, timing not included */
    if(cuda_dev_id >= 0) {
        sptCudaSetDevice(cuda_dev_id);
        // sptAssert(sptCudaMTTKRP(&X, U, mats_order, mode, impl_num) == 0);
        sptAssert(sptCudaMTTKRPOneKernel(&X, U, mats_order, mode, impl_num) == 0);

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
    }

    
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);
    for(int it=0; it<niters; ++it) {
        // sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
        if(cuda_dev_id >= 0) {
            sptCudaSetDevice(cuda_dev_id);
            // sptAssert(sptCudaMTTKRP(&X, U, mats_order, mode, impl_num) == 0);
            sptAssert(sptCudaMTTKRPOneKernel(&X, U, mats_order, mode, impl_num) == 0);
        }
    }
    sptStopTimer(timer);

    if (cuda_dev_id >=0) {
        double aver_time = sptPrintAverageElapsedTime(timer, niters, "GPU SpTns MTTKRP");
    }

    sptFreeTimer(timer);
    for(sptIndex m=0; m<nmodes; ++m) {
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

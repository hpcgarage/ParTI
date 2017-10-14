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
#include "../src/sptensor/hicoo/hicoo.h"

int main(int argc, char * const argv[]) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptRankMatrix U;
    sptSparseTensorHiCOO X, Y;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    sptElementIndex sc_bits;

    sptIndex mode = 0;
    sptIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int tk = 1;
    printf("niters: %d\n", niters);

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"mode", required_argument, 0, 'm'},
            {"impl-num", optional_argument, 0, 'p'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:d:r:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            sptAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            break;
        case 'b':
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sk_bits);
            break;
        case 'c':
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sc_bits);
            break;
        case 'm':
            sscanf(optarg, "%"SPT_PF_INDEX, &mode);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"SPT_PF_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        default:
            abort();
        }
    }

    // if(optind > argc) {
    if(argc < 2) {
        printf("Usage: %s [options] \n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
        printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
        printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
        printf("         -m MODE, --mode=MODE\n");
        printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -r RANK\n");
        printf("         -t TK, --tk=TK\n");
        printf("\n");
        return 1;
    }

    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Convert to HiCOO tensor */
    sptAssert(sptSparseTensorToHiCOO(&X, &tsr, sb_bits, sk_bits, sc_bits) == 0);
    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&X, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&X, stdout) == 0);

    sptIndex nmodes = X.nmodes;
    sptIndex * ndims = (sptIndex*)malloc(nmodes * sizeof(*ndims));
    ndims[mode] = R;
    sptAssert(sptNewSparseTensorHiCOO_NoNnz(&Y, nmodes, ndims, sb_bits, sk_bits, sc_bits) == 0);
    free(ndims);


    sptAssert(sptNewRankMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptConstantRankMatrix(&U, 0) == 0);
    sptIndex stride = U.stride;
    // sptAssert(sptDumpMatrix(&U, stdout) == 0);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        sptAssert(sptTTMHiCOO_MatrixTiling(&Y, &X, &U, mode) == 0);
    } /* else if(cuda_dev_id == -1) {
        printf("tk: %d\n", tk);
        sptAssert(sptOmpTTMHiCOO_MatrixTiling(&Y, &X, &U, mode, tk) == 0);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, impl_num) == 0);
    } */

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            nthreads = 1;
            sptAssert(sptTTMHiCOO_MatrixTiling(&Y, &X, &U, mode) == 0);
        } /* else if(cuda_dev_id == -1) {
            printf("tk: %d\n", tk);
            sptAssert(sptOmpTTMHiCOO_MatrixTiling(&Y, &X, &U, mode, tk) == 0);
        } else {
            sptCudaSetDevice(cuda_dev_id);
            sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, impl_num) == 0);
        } */
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns TTM");
    sptFreeTimer(timer);

    if(fo != NULL) {
        sptAssert(sptDumpSparseTensorHiCOO(&Y, fo) == 0);
        fclose(fo);
    }

    sptFreeRankMatrix(&U);
    sptFreeSparseTensorHiCOO(&X);
    sptFreeSparseTensorHiCOO(&Y);

    return 0;
}

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

void print_usage(int argc, char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -m MODE, --mode=MODE\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -l TB, --tb=TB\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptMatrix ** U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    sptElementIndex sc_bits;

    sptIndex mode = 0;
    sptIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int renumber = 0;
    int tk = 1;
    int tb = 1;
    printf("niters: %d\n", niters);
    int retval;

    if(argc <= 3) { // #Required arguments
        print_usage(argc, argv);
        exit(1);
    }

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"mode", required_argument, 0, 'm'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'l'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:e:d:r:t:l:", long_options, &option_index);
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
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 'c':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sc_bits);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'l':
            sscanf(optarg, "%d", &tb);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argc, argv);
            exit(1);
        }
    }


    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    // sptSparseTensorSortIndex(&tsr, 1);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Renumber the input tensor */
    if (renumber == 1) {
        sptIndex ** map_inds = (sptIndex **)malloc(tsr.nmodes * sizeof *map_inds);
        spt_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            map_inds[m] = (sptIndex *)malloc(tsr.ndims[m] * sizeof (sptIndex));
            spt_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(sptIndex i = 0; i < tsr.ndims[m]; ++i) 
                map_inds[m][i] = i;
        }
        /* Set randomly renumbering */
        sptGetRandomShuffledIndices(&tsr, map_inds);

        sptSparseTensorShuffleIndices(&tsr, map_inds);

        // sptSparseTensorSortIndex(&tsr, 1);
        // printf("map_inds:\n");
        // for(sptIndex m = 0; m < tsr.nmodes; ++m) {
        //     sptDumpIndexArray(map_inds[m], stdout);
        // }
        // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }

    sptTimer convert_timer;
    sptNewTimer(&convert_timer, 0);
    sptStartTimer(convert_timer);

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits) == 0);
    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);
    if (max_nnzb > 1024 && cuda_dev_id >= 0 ) {
        printf("Too many nnzs per block. \n");
        return -1;
    }

    sptStopTimer(convert_timer);
    sptPrintElapsedTime(convert_timer, "Convert HiCOO");
    sptFreeTimer(convert_timer);

    sptIndex nmodes = hitsr.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], tsr.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], hitsr.ndims[m], R) == 0);
      // sptAssert(sptConstantMatrix(U[m], 1) == 0);
      sptAssert(sptRandomizeMatrix(U[m], hitsr.ndims[m], R) == 0);
      if(hitsr.ndims[m] > max_ndims)
        max_ndims = hitsr.ndims[m];
      // sptAssert(sptDumpMatrix(U[m], stdout) == 0);
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    // sptAssert(sptDumpMatrix(U[nmodes], stdout) == 0);


    sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));
    mats_order[0] = mode;
    for(sptIndex i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;
    // printf("mats_order:\n");
    // sptDumpIndexArray(mats_order, nmodes, stdout);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        sptAssert(sptMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
    } else if(cuda_dev_id == -1) {
        printf("tk: %d, tb: %d\n", tk, tb);
        sptAssert(sptOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, tk, tb) == 0);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, max_nnzb, impl_num) == 0);
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            nthreads = 1;
            sptAssert(sptMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            sptAssert(sptOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, tk, tb) == 0);
        } else {
            sptCudaSetDevice(cuda_dev_id);
            sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, max_nnzb, impl_num) == 0);
        } 
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
    sptFreeTimer(timer);

    if(fo != NULL) {
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }


    for(sptIndex m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    free(mats_order);
    sptFreeMatrix(U[nmodes]);
    free(U);
    sptFreeSparseTensorHiCOO(&hitsr);

    return 0;
}

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
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"
#include "../src/sptensor/hicoo/hicoo.h"


void print_usage(char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -h TB, --tb=TB\n");
    printf("\n");
}


int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptSparseTensorHiCOO hitsr;
    sptRankKruskalTensor ktensor;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;
    sptElementIndex sc_bits;

    sptIndex R = 16;
    int cuda_dev_id = -2;
    // int nloops = 1; // 5
    sptIndex niters = 5; // 50
    double tol = 1e-5;
    int impl_num = 0;
    int tk = 1;
    int tb = 1;

    if(argc < 2) {
        print_usage(argv);
        exit(1);
    }


    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"impl-num", optional_argument, 0, 'p'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'h'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:o:b:k:c:p:d:r:t:h:", long_options, &option_index);
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
        case 'p':
            sscanf(optarg, "%d", &impl_num);
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
        case 'h':
            sscanf(optarg, "%d", &tb);
            break;
        default:
            abort();
        }
    }

    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits, tk) == 0);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    sptIndex nmodes = hitsr.nmodes;
    sptNewRankKruskalTensor(&ktensor, nmodes, tsr.ndims, R);
    sptFreeSparseTensor(&tsr);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        tk = 1;
        sptAssert(sptCpdAlsHiCOO(&hitsr, R, niters, tol, &ktensor) == 0);
    } else if(cuda_dev_id == -1) {
        omp_set_num_threads(tk);
        #pragma omp parallel
        {
            tk = omp_get_num_threads();
        }
        printf("tk: %d, tb: %d\n", tk, tb);
        sptAssert(sptOmpCpdAlsHiCOO(&hitsr, R, niters, tol, tk, tb, &ktensor) == 0);
    }

    // for(int it=0; it<nloops; ++it) {
    // }

    if(fo != NULL) {
        // Dump ktensor to files
        sptAssert( sptDumpRankKruskalTensor(&ktensor, fo) == 0 );
        fclose(fo);
    }

    sptFreeSparseTensorHiCOO(&hitsr);
    sptFreeRankKruskalTensor(&ktensor);

    return 0;
}

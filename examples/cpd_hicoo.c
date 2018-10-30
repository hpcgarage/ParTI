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
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (required)\n");
    printf("         -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (required)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (CPD rank, 16:default)\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         --help\n");
    printf("\n");
}


int main(int argc, char ** argv) {
    printf("CPD HiCOO: \n");

    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptSparseTensorHiCOO hitsr;
    sptRankKruskalTensor ktensor;
    sptElementIndex sb_bits;
    sptElementIndex sk_bits;

    sptIndex R = 16;
    int dev_id = -2;
    int nloops = 5;
    sptIndex niters = 5; // 50
    double tol = 1e-5;
    int nt = 1;

    if(argc < 5) { // #Required arguments
        print_usage(argv);
        exit(1);
    }


    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"output", optional_argument, 0, 'o'},
            {"dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nt", optional_argument, 0, 't'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:b:k:o:d:r:t:", long_options, &option_index);
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
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("dev_id: %d\n", dev_id);

    /* A sorting included in load tensor */
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptTimer convert_timer;
    sptNewTimer(&convert_timer, 0);
    sptStartTimer(convert_timer);

    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, nt) == 0);

    sptStopTimer(convert_timer);
    sptPrintElapsedTime(convert_timer, "Convert HiCOO");
    sptFreeTimer(convert_timer);

    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    sptIndex nmodes = hitsr.nmodes;
    sptNewRankKruskalTensor(&ktensor, nmodes, tsr.ndims, R);
    sptFreeSparseTensor(&tsr);

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        nt = 1;
        sptAssert(sptCpdAlsHiCOO(&hitsr, R, niters, tol, &ktensor) == 0);
    } else if(dev_id == -1) {
        omp_set_num_threads(nt);
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
        printf("nt: %d \n", nt);
        sptAssert(sptOmpCpdAlsHiCOO(&hitsr, R, niters, tol, nt, &ktensor) == 0);
    }


    if(fo != NULL) {
        // Dump ktensor to files
        sptAssert( sptDumpRankKruskalTensor(&ktensor, fo) == 0 );
        fclose(fo);
    }

    sptFreeSparseTensorHiCOO(&hitsr);
    sptFreeRankKruskalTensor(&ktensor);

    return 0;
}

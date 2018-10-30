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
#ifdef PARTI_USE_OPENMP
#include <omp.h>
#endif

void print_usage(int argc, char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=CUDA_DEV_ID (>=0:GPU device id)\n");
    printf("         -r RANK (CPD rank, 16:default)\n");
    printf("         GPU options: \n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM (11, 12, 15, where 15 should be the best case)\n");
    printf("         --help\n");
    printf("\n");
}


int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;
    sptIndex R = 16;
    sptIndex niters = 5; // 50
    double tol = 1e-5;
    sptKruskalTensor ktensor;
    int nloops = 1;
    int cuda_dev_id = -2;
    int impl_num = 0;

    if(argc < 2) {
        print_usage(argc, argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:p:d:r:", long_options, &option_index);
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

    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&X, stdout);
    // sptDumpSparseTensor(&X, 0, stdout);

    sptIndex nmodes = X.nmodes;
    sptNewKruskalTensor(&ktensor, nmodes, X.ndims, R);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id >= 0) {
         sptCudaSetDevice(cuda_dev_id);
         sptAssert(sptCudaCpdAls(&X, R, niters, tol, &ktensor) == 0);
    } 


    if(fo != NULL) {
        // Dump ktensor to files
        sptAssert( sptDumpKruskalTensor(&ktensor, fo) == 0 );
        fclose(fo);
    }

    sptFreeSparseTensor(&X);
    sptFreeKruskalTensor(&ktensor);

    return 0;
}

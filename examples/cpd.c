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
#include "../src/sptensor/hicoo/hicoo.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (CPD rank, 16:default)\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         -u use_reduce, --ur=use_reduce (use privatization or not)\n");
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
    int nloops = 5;
    int dev_id = -2;
    int nthreads = 1;
    int use_reduce = 0;

    if(argc < 2) {
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", optional_argument, 0, 'o'},
            {"dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nt", optional_argument, 0, 't'},
            {"use-reduce", optional_argument, 0, 'u'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:d:r:t:u:", long_options, &option_index);
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
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u"PARTI_SCN_INDEX, &R);
            break;
        case 'u':
            sscanf(optarg, "%d", &use_reduce);
            break;
        case 't':
            sscanf(optarg, "%d", &nthreads);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("dev_id: %d\n", dev_id);

    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&X, stdout);
    // sptDumpSparseTensor(&X, 0, stdout);  

    sptIndex nmodes = X.nmodes;
    sptNewKruskalTensor(&ktensor, nmodes, X.ndims, R);

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        nthreads = 1;
        sptAssert(sptCpdAls(&X, R, niters, tol, &ktensor) == 0);
    } else if(dev_id == -1) {
        omp_set_num_threads(nthreads);
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        printf("use_reduce: %d\n", use_reduce);
        sptAssert(sptOmpCpdAls(&X, R, niters, tol, nthreads, use_reduce, &ktensor) == 0);
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

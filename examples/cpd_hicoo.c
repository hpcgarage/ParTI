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
#include <omp.h>


void print_usage(char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -n NITERS_RENUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -a balanced\n");
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
    sptElementIndex sc_bits;

    sptIndex R = 16;
    int cuda_dev_id = -2;
    // int nloops = 1; // 5
    sptIndex niters = 5; //5; // 50
    double tol = 1e-5;
    int renumber = 0;
    int niters_renum = 3;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */
    int tk = 1;
    int balanced = 0;

    if(argc < 5) { // #Required arguments
        print_usage(argv);
        exit(1);
    }


    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"balanced", optional_argument, 0, 'a'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:b:k:c:o:e:n:d:r:t:a:", long_options, &option_index);
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
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
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
        case 'a':
            sscanf(optarg, "%d", &balanced);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);
    printf("balanced: %d\n", balanced);

    /* A sorting included in load tensor */
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Renumber the input tensor */
    sptIndex ** map_inds;
    if (renumber > 0) {
        map_inds = (sptIndex **)malloc(tsr.nmodes * sizeof *map_inds);
        spt_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            map_inds[m] = (sptIndex *)malloc(tsr.ndims[m] * sizeof (sptIndex));
            spt_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(sptIndex i = 0; i < tsr.ndims[m]; ++i) 
                map_inds[m][i] = i;
        }

        sptTimer renumber_timer;
        sptNewTimer(&renumber_timer, 0);
        sptStartTimer(renumber_timer);

        if ( renumber == 1 || renumber == 2) { /* Set the Lexi-order or BFS-like renumbering */
            sptIndexRenumber(&tsr, map_inds, renumber, niters_renum, tk);
        }
        if ( renumber == 3) { /* Set randomly renumbering */
            sptGetRandomShuffledIndices(&tsr, map_inds);
        }
        fflush(stdout);

        sptStopTimer(renumber_timer);
        sptPrintElapsedTime(renumber_timer, "Renumbering");
        sptFreeTimer(renumber_timer);

        sptTimer shuffle_timer;
        sptNewTimer(&shuffle_timer, 0);
        sptStartTimer(shuffle_timer);

        sptSparseTensorShuffleIndices(&tsr, map_inds);

        sptStopTimer(shuffle_timer);
        sptPrintElapsedTime(shuffle_timer, "Shuffling time");
        sptFreeTimer(shuffle_timer);
        printf("\n");


        // sptSparseTensorSortIndex(&tsr, 1);   // debug purpose only
        // FILE * debug_fp = fopen("new.txt", "w");
        // sptAssert(sptDumpSparseTensor(&tsr, 0, debug_fp) == 0);
        // fprintf(stdout, "\nmap_inds:\n");
        // for(sptIndex m = 0; m < tsr.nmodes; ++m) {
        //     sptDumpIndexArray(map_inds[m], tsr.ndims[m], stdout);
        // }
        // fclose(debug_fp);
    }

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptTimer convert_timer;
    sptNewTimer(&convert_timer, 0);
    sptStartTimer(convert_timer);

    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, tk) == 0);

    sptStopTimer(convert_timer);
    sptPrintElapsedTime(convert_timer, "Convert HiCOO");
    sptFreeTimer(convert_timer);

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
        printf("tk: %d\n", tk);
        sptAssert(sptOmpCpdAlsHiCOO(&hitsr, R, niters, tol, tk, balanced, &ktensor) == 0);
    }

    // for(int it=0; it<nloops; ++it) {
    // }

    if(fo != NULL) {
        // Dump ktensor to files
        if (renumber > 0) {
            sptRankKruskalTensorInverseShuffleIndices(&ktensor, map_inds);
        }
        sptAssert( sptDumpRankKruskalTensor(&ktensor, fo) == 0 );
        fclose(fo);
    }

    if (renumber > 0) {
        for(sptIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }
    sptFreeSparseTensorHiCOO(&hitsr);
    sptFreeRankKruskalTensor(&ktensor);

    return 0;
}

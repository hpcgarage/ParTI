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

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         Reordering options: \n");
    printf("         -e RELABEL, --relabel=RELABEL (0:no-relabeling,default; 1:relabel with Lexi-order; 2:relabel with BFS-MCS; 3:randomly relabel)\n");
    printf("         -n NITERS_RENUM (default: 3, required when -e 1)\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;
    /* relabel:
     * = 0 : no relabel.
     * = 1 : relabel with Lexi-order
     * = 2 : relabel with BFS-like
     * = 3 : randomly relabel, specify niters_renum.
     */
    int relabel = 0;
    int niters_renum = 3;
    int nt = 1;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(-1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", optional_argument, 0, 'o'},
            {"relabel", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"nt", optional_argument, 0, 't'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:e:n:t:", long_options, &option_index);
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
            fo = fopen(optarg, "aw");
            sptAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'e':
            sscanf(optarg, "%d", &relabel);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(-1);
        }
    }

    printf("relabel: %d\n", relabel);
    if (relabel == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* Load a sparse tensor from file as it is */
    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);

    /* relabel the input tensor */
    sptIndex ** map_inds;
    map_inds = (sptIndex **)malloc(X.nmodes * sizeof *map_inds);
    spt_CheckOSError(!map_inds, "MTTKRP HiCOO");
    for(sptIndex m = 0; m < X.nmodes; ++m) {
        map_inds[m] = (sptIndex *)malloc(X.ndims[m] * sizeof (sptIndex));
        spt_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
        for(sptIndex i = 0; i < X.ndims[m]; ++i) 
            map_inds[m][i] = i;
    }

    sptTimer renumber_timer;
    sptNewTimer(&renumber_timer, 0);
    sptStartTimer(renumber_timer);

    if ( relabel == 1 || relabel == 2) { /* Set the Lexi-order or BFS-like renumbering */
        sptIndexRenumber(&X, map_inds, relabel, niters_renum, nt);
    }
    if ( relabel == 3) { /* Set randomly renumbering */
        sptGetRandomShuffledIndices(&X, map_inds);
    }

    sptStopTimer(renumber_timer);
    sptPrintElapsedTime(renumber_timer, "Renumbering");
    sptFreeTimer(renumber_timer);

    sptTimer shuffle_timer;
    sptNewTimer(&shuffle_timer, 0);
    sptStartTimer(shuffle_timer);

    sptSparseTensorShuffleIndices(&X, map_inds);

    sptStopTimer(shuffle_timer);
    sptPrintElapsedTime(shuffle_timer, "Shuffling time");
    sptFreeTimer(shuffle_timer);
    printf("\n");


    if(fo != NULL) {
        sptAssert(sptDumpSparseTensor(&X, 1, fo) == 0);
        fclose(fo);
    }
    for(sptIndex m = 0; m < X.nmodes; ++m) {
        free(map_inds[m]);
    }
    free(map_inds);
    sptFreeSparseTensor(&X);

    return 0;
}

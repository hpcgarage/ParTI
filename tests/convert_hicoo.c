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

int main(int argc, char * const argv[]) {
    FILE *fi, *fo;
    sptSparseTensor tsr;
    sptSparseTensorHiCOO hitsr;
    const sptElementIndex sb;
    const sptBlockIndex sk;
    const sptBlockNnzIndex sc;


    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:b:k:c:", long_options, &option_index);
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
            sscanf(optarg, "%"SPT_PF_ELEMENTINDEX, &sb);
            break;
        case 'k':
            sscanf(optarg, "%"SPT_PF_NNZINDEX, &sk);
            break;
        case 'c':
            sscanf(optarg, "%"SPT_PF_BLOCKNNZINDEX, &sc);
            break;
        default:
            abort();
        }
    }

    if(optind > argc) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -b BLOCKSIZE, --blocksize=BLOCKSIZE\n");
        printf("         -k KERNELSIZE, --kernelsize=KERNELSIZE\n");
        printf("         -c CHUNKSIZE, --chunksize=CHUNKSIZE\n");
        printf("\n");
        return 1;
    }

    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&tsr, stdout);
    sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    sptAssert(sptSparseTensorToHiCOO(&hitsr, &tsr, sb, sk, sc) == 0);
    sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);
    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);

    sptAssert(sptDumpSparseTensorHiCOO(&hitsr, 0, fo) == 0);
    fclose(fo);

    sptFreeSparseTensorHiCOO(&hitsr);

    return 0;
}

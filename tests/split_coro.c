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
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"


int main(int argc, char *argv[]) {
    FILE *fi;
    sptSparseTensor tsr;

    if(argc < 3) {
        printf("Usage: %s input size1 [size2 ...]\n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    sptAssert(fi != NULL);
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);

    sptAssert((int) tsr.nmodes + 2 == argc);

    size_t *sizes = malloc(tsr.nmodes * sizeof (size_t));
    size_t i;
    for(i = 0; i < tsr.nmodes; ++i) {
        sizes[i] = atoi(argv[i+2]);
    }

    size_t *inds_low = malloc(tsr.nmodes * sizeof (size_t));
    size_t *inds_high = malloc(tsr.nmodes * sizeof (size_t));

    printf("Splitting using API 'Split Coroutine', max size [");
    print_inds(sizes, tsr.nmodes, 0, stdout);
    printf("].\n\n");

    spt_SplitHandle split_handle;
    sptAssert(spt_StartSplitSparseTensor(&split_handle, &tsr, sizes) == 0);

    for(i = 1; ; ++i) {
        sptSparseTensor subtsr;
        int result = spt_SplitSparseTensor(&subtsr, inds_low, inds_high, split_handle);
        if(result == SPTERR_NO_MORE) { break; }
        sptAssert(result == 0);

        printf("Printing split #%zu:\n", i);
        printf("Index: [");
        print_inds(inds_low, subtsr.nmodes, 1, stdout);
        printf("] .. [");
        print_inds(inds_high, subtsr.nmodes, 1, stdout);
        printf("].\n");
        sptDumpSparseTensor(&subtsr, 1, stdout);
        printf("\n");
        fflush(stdout);

        sptFreeSparseTensor(&subtsr);
    }

    spt_FinishSplitSparseTensor(split_handle);
    free(inds_high);
    free(inds_low);
    free(sizes);
    sptFreeSparseTensor(&tsr);

    return 0;

}

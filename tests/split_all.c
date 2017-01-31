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

static void print_inds(const size_t array[], size_t length, size_t start_index) {
    if(length == 0) {
        return;
    }
    printf("%zu", array[0] + start_index);
    size_t i;
    for(i = 1; i < length; ++i) {
        printf(", %zu", array[i] + start_index);
    }
}

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

    printf("Splitting using API 'GetAllSplits', max size [");
    print_inds(sizes, tsr.nmodes, 0);
    printf("].\n\n");

    spt_SplitResult *splits;
    size_t nsplits;
    sptAssert(spt_SparseTensorGetAllSplits(&splits, &nsplits, &tsr, sizes, 1) == 0);

    spt_SplitResult *split_i = splits;
    for(i = 0; i < nsplits; ++i) {
        printf("Printing split #%zu of %zu:\n", i + 1, nsplits);
        printf("Index: [");
        print_inds(split_i->inds_low, split_i->tensor.nmodes, 1);
        printf("] .. [");
        print_inds(split_i->inds_high, split_i->tensor.nmodes, 1);
        printf("].\n");
        sptDumpSparseTensor(&split_i->tensor, 1, stdout);
        printf("\n");
        fflush(stdout);
        split_i = split_i->next;
    }

    spt_SparseTensorFreeAllSplits(splits);
    free(sizes);
    sptFreeSparseTensor(&tsr);

    return 0;

}

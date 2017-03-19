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
        printf("Usage: %s input mode step \n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    sptAssert(fi != NULL);
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);

    sptAssert(4 == argc);

    size_t mode = atoi(argv[2]);
    size_t step = atoi(argv[3]);

    printf("step: %lu, mode: %lu\n", step, mode);

    size_t const nmodes = tsr.nmodes;
    size_t * mode_order = (size_t *)malloc(nmodes * sizeof(size_t));
    mode_order[0] = mode;
    for(size_t i=1; i<nmodes; ++i)
        mode_order[i] = (mode+i) % nmodes;
    printf("mode_order:\n");
    spt_DumpArray(mode_order, nmodes, 0, stdout);

    sptSparseTensorSortIndexCustomOrder(&tsr, mode_order);  // tsr sorted from mode-0, ..., N-1.
    sptDumpSparseTensor(&tsr, 0, stdout);


    spt_SplitResult *splits;
    size_t nsplits;
    sptAssert(spt_CoarseSplitSparseTensorAll(&splits, &nsplits, step, mode, &tsr) == 0);
    spt_SparseTensorDumpAllSplits(splits, nsplits, stdout);

    // spt_SparseTensorFreeAllSplits(splits, nsplits);
    sptFreeSparseTensor(&tsr);
    free(splits);
    free(mode_order);

    return 0;

}

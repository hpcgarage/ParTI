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
        printf("Usage: %s input step1 [step2 ...]\n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    sptAssert(fi != NULL);
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);

    sptAssert((int) tsr.nmodes + 2 == argc);

    size_t *steps = malloc(tsr.nmodes * sizeof (size_t));
    size_t i;
    for(i = 0; i < tsr.nmodes; ++i) {
        steps[i] = atoi(argv[i+2]);
    }

    printf("Splitting using API 'IndexSplit', max index step [");
    spt_DumpArray(steps, tsr.nmodes, 0, stdout);
    printf("].\n\n");

    spt_SplitResult *splits;
    size_t nsplits;
    sptAssert(spt_SparseTensorGetAllSplits(&splits, &nsplits, &tsr, NULL, steps, 1) == 0);
    spt_SparseTensorDumpAllSplits(splits, nsplits, stdout);

    spt_SparseTensorFreeAllSplits(splits, nsplits);
    free(steps);
    sptFreeSparseTensor(&tsr);

    return 0;

}

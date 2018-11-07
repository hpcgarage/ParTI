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
#include <ParTI.h>

int main(int argc, char *argv[]) {
    FILE *fi, *fo;
    sptSparseTensor tsr;

    if(argc != 3) {
        printf("Usage: %s input location\n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    sptAssert(fi != NULL);
    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);

    sptSparseTensorStatus(&tsr, stdout);


    sptNnzIndex loc = atol(argv[2]);
    sptIndex const nmodes = tsr.nmodes;

    printf("Current entry: ( ");
    for(sptIndex m = 0; m < nmodes - 1; ++m) {
        printf("%" PARTI_PRI_INDEX ", ", tsr.inds[m].data[loc]);
    }
    printf("%" PARTI_PRI_INDEX " ) ", tsr.inds[nmodes-1].data[loc]);
    printf("%" PARTI_PRI_VALUE "\n", tsr.values.data[loc]);

    sptFreeSparseTensor(&tsr);

    return 0;
}

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

int main(int argc, char const *argv[]) {
    FILE *fi, *fo;
    sptIndex mode = 0;
    sptSparseTensor a;
    sptSemiSparseTensor b;
    sptSparseTensor c;

    if(argc != 4) {
        printf("Usage %s in out mode\n\n", argv[0]);
        return 1;
    }

    sscanf(argv[3], "%zu", &mode);

    fi = fopen(argv[1], "r");
    sptAssert(fi);
    sptAssert(sptLoadSparseTensor(&a, 1, fi) == 0);
    fclose(fi);

    sptAssert(sptSparseTensorToSemiSparseTensor(&b, &a, mode) == 0);
    sptFreeSparseTensor(&a);
    sptAssert(sptSemiSparseTensorToSparseTensor(&c, &b, 1e-6) == 0);
    sptFreeSemiSparseTensor(&b);

    fo = fopen(argv[2], "w");
    sptAssert(sptDumpSparseTensor(&c, 1, fo) == 0);
    fclose(fo);

    sptFreeSparseTensor(&c);

    return 0;
}

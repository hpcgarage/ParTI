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
    FILE *fa, *fb, *fo;
    sptSparseTensor a, b, out;

    if(argc != 4) {
        printf("Usage: %s a b out\n\n", argv[0]);
        return 1;
    }

    fa = fopen(argv[1], "r");
    sptAssert(fa != NULL);
    sptAssert(sptLoadSparseTensor(&a, 1, fa) == 0);
    fclose(fa);

    fb = fopen(argv[2], "r");
    sptAssert(fb != NULL);
    sptAssert(sptLoadSparseTensor(&b, 1, fb) == 0);
    fclose(fb);

    sptAssert(sptSparseTensorKhatriRaoMul(&out, &a, &b) == 0);

    fo = fopen(argv[3], "w");
    sptAssert(fo != NULL);
    sptAssert(sptDumpSparseTensor(&out, 1, fo) == 0);
    fclose(fo);

    return 0;
}

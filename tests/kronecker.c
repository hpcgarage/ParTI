/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <stdio.h>
#include <SpTOL.h>

int main(int argc, char *argv[]) {
    FILE *fa, *fb, *fo;
    sptSparseTensor a, b, out;
    
    if(argc != 4) {
        printf("Usage: %s a b out\n\n", argv[0]);
        return 1;
    }

    fa = fopen(argv[1], "r");
    assert(fa != NULL);
    assert(sptLoadSparseTensor(&a, 1, fa) == 0);
    fclose(fa);

    fb = fopen(argv[2], "r");
    assert(fb != NULL);
    assert(sptLoadSparseTensor(&b, 1, fb) == 0);
    fclose(fb);

    assert(sptSparseTensorKroneckerMul(&out, &a, &b) == 0);

    fo = fopen(argv[3], "w");
    assert(fo != NULL);
    assert(sptDumpSparseTensor(&out, 1, fo) == 0);
    fclose(fo);

    return 0;
}

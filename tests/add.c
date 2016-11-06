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
#include <stdlib.h>
#include <SpTOL.h>
#include "timer.h"

int main(int argc, char *argv[]) {
    FILE *fa, *fb, *fo;
    sptSparseTensor a, b, c;

    if(argc != 6) {
        printf("Usage: %s nthreads niters a b out\n\n", argv[0]);
        return 1;
    }

    int nthreads = atoi(argv[1]);
    int niters = atoi(argv[2]);
    assert(nthreads >=1);
    assert(niters >= 1);

    fa = fopen(argv[3], "r");
    assert(fa != NULL);
    assert(sptLoadSparseTensor(&a, 1, fa) == 0);
    fclose(fa);

    fb = fopen(argv[4], "r");
    assert(fb != NULL);
    assert(sptLoadSparseTensor(&b, 1, fb) == 0);
    fclose(fb);

    sptSparseTensorSortIndex(&a);
    sptSparseTensorSortIndex(&b);

    // Timer add_timer;

    // timer_fstart(&add_timer);
    if(nthreads == 1) {
        for(int i=0; i<niters; ++i) {
            assert(sptSparseTensorAdd(&c, &a, &b) == 0);
        }
    } else {
        for(int i=0; i<niters; ++i) {
            assert(sptSparseTensorAddOMP(&a, &b, nthreads) == 0);
        }
    }
    // timer_stop(&add_timer);
    // double add_sec = add_timer.seconds / niters;
    // printf("Element-wise add: %.3f sec\n", add_sec);

    fo = fopen(argv[5], "w");
    assert(fo != NULL);
    assert(sptDumpSparseTensor(&c, 1, fo) == 0);
    fclose(fo);

    return 0;
}

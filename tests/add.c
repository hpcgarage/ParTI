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

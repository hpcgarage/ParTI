#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>
#include "timer.h"

int main(int argc, char *argv[]) {
    FILE *fa, *fb, *fo;
    sptSparseTensor a, b;
    
    if(argc != 5) {
        printf("Usage: %s niters a b out\n\n", argv[0]);
        return 1;
    }

    int niters = atoi(argv[1]);

    fa = fopen(argv[2], "r");
    assert(fa != NULL);
    assert(sptLoadSparseTensor(&a, fa) == 0);
    fclose(fa);

    fb = fopen(argv[3], "r");
    assert(fb != NULL);
    assert(sptLoadSparseTensor(&b, fb) == 0);
    fclose(fb);

    Timer add_timer;

    // timer_fstart(&add_timer);
    for(int i=0; i<niters; ++i) {
        assert(sptSparseTensorAdd(&a, &b) == 0);
    }
    // timer_stop(&add_timer);
    // double add_sec = add_timer.seconds / niters;
    // printf("Element-wise add: %.3f sec\n", add_sec);

    fo = fopen(argv[4], "w");
    assert(fo != NULL);
    assert(sptDumpSparseTensor(&a, fo) == 0);
    fclose(fo);

    return 0;
}

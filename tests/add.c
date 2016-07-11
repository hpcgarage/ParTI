#include <assert.h>
#include <stdio.h>
#include <SpTOL.h>

int main(int argc, char *argv[]) {
    FILE *fa, *fb, *fo;
    sptSparseTensor a, b;
    
    if(argc != 4) {
        printf("Usage: %s a b out\n\n", argv[0]);
        return 1;
    }

    fa = fopen(argv[1], "r");
    assert(fa != NULL);
    assert(sptLoadSparseTensor(&a, fa) == 0);
    fclose(fa);

    fb = fopen(argv[2], "r");
    assert(fb != NULL);
    assert(sptLoadSparseTensor(&b, fb) == 0);
    fclose(fb);

    assert(sptSparseTensorAdd(&a, &b) == 0);

    fo = fopen(argv[3], "w");
    assert(fo != NULL);
    assert(sptDumpSparseTensor(&a, fo) == 0);
    fclose(fo);

    return 0;
}

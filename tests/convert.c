#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char const *argv[]) {
    FILE *fi, *fo;
    sptSparseTensor a;
    sptSemiSparseTensor b;
    sptSparseTensor c;

    if(argc != 3) {
        printf("Usage %s in out\n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    assert(fi);
    assert(sptLoadSparseTensor(&a, fi) == 0);
    fclose(fi);

    assert(sptSparseTensorToSemiSparseTensor(&b, &a, 0) == 0);
    assert(sptSemiSparseTensorToSparseTensor(&c, &b) == 0);

    fo = fopen(argv[2], "w");
    assert(sptDumpSparseTensor(&c, fo) == 0);
    fclose(fo);

    return 0;
}

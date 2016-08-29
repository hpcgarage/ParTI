#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <SpTOL.h>

int main(int argc, char const *argv[]) {
    FILE *fi, *fo;
    size_t mode = 0;
    sptSparseTensor a;
    sptSemiSparseTensor b;
    sptSparseTensor c;

    if(argc != 4) {
        printf("Usage %s in out mode\n\n", argv[0]);
        return 1;
    }

    sscanf(argv[3], "%zu", &mode);

    fi = fopen(argv[1], "r");
    assert(fi);
    assert(sptLoadSparseTensor(&a, 1, fi) == 0);
    fclose(fi);

    assert(sptSparseTensorToSemiSparseTensor(&b, &a, mode) == 0);
    sptFreeSparseTensor(&a);
    assert(sptSemiSparseTensorToSparseTensor(&c, &b, 1e-6) == 0);
    sptFreeSemiSparseTensor(&b);

    fo = fopen(argv[2], "w");
    assert(sptDumpSparseTensor(&c, 1, fo) == 0);
    fclose(fo);

    sptFreeSparseTensor(&c);

    return 0;
}

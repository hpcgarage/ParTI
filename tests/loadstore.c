#include <assert.h>
#include <stdio.h>
#include <SpTOL.h>

int main(int argc, char *argv[]) {
    FILE *fi, *fo;
    sptSparseTensor tsr;

    if(argc != 3) {
        printf("Usage: %s input output\n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    assert(fi != NULL);
    assert(sptLoadSparseTensor(&tsr, fi) == 0);
    fclose(fi);

    fo = fopen(argv[2], "w");
    assert(fo != NULL);
    assert(sptDumpSparseTensor(&tsr, fo) == 0);
    fclose(fo);

    sptFreeSparseTensor(&tsr);

    return 0;
}

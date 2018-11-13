/*
    Internal code for ParTI!
    (c) Sam Bliss, 2018, all rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ParTI.h>
#include "../src/error/error.h"

static int do_ttm(sptSparseTensor *X, sptMatrix *U, sptIndex mode, int cuda_dev_id) {
    sptSemiSparseTensor Y;
    int result;
    if(cuda_dev_id == -2) {
        result = sptSparseTensorMulMatrix(&Y, X, U, mode);
    } else if(cuda_dev_id == -1) {
        result = sptOmpSparseTensorMulMatrix(&Y, X, U, mode);
    } else {
        result = sptCudaSparseTensorMulMatrix(&Y, X, U, mode);
    }
    spt_CheckError(result, "do_ttm", NULL);
    sptFreeSemiSparseTensor(&Y);
    return 0;
}

static int spt_LoadMatrixTranspose(sptMatrix *X, FILE *f) {
    int result = 0;
    sptIndex nmodes, nrows, ncols;
    result = fscanf(f, "%"PARTI_SCN_INDEX"%"PARTI_SCN_INDEX"%"PARTI_SCN_INDEX, &nmodes, &ncols, &nrows);
    spt_CheckOSError(result < 3, "LoadMtx");
    if(nmodes != 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "LoadMtx", "nmodes != 2");
    }
    result = sptNewMatrix(X, nrows, ncols);
    spt_CheckError(result, "LoadMtx", NULL);
    memset(X->values, 0, X->nrows * X->stride * sizeof (sptValue));
    sptIndex i, j;
    for(i = 0; i < X->ncols; ++i) {
        for(j = 0; j < X->nrows; ++j) {
            double value;
            result = fscanf(f, "%lf", &value);
            spt_CheckOSError(result < 1, "LoadMtx");
            X->values[j * X->stride + i] = value;
        }
    }
    return 0;
}

int main(int argc, char const *argv[]) {
    FILE *fX, *fU;
    sptSparseTensor X;
    sptMatrix U;
    sptIndex mode = 0;
    int cuda_dev_id = -2;
    int result = 0;

    if(argc != 5) {
        fprintf(stderr, "Usage: %s X U mode cuda_dev_id\n", argv[0]);
        return 0;
    }

    fX = fopen(argv[1], "r");
    if(!fX) {
        spt_CheckError(SPTERR_OS_ERROR, "fopen", NULL);
    }
    result = sptLoadSparseTensor(&X, 1, fX);
    spt_CheckError(result, "main", NULL);
    fclose(fX);

    fU = fopen(argv[2], "r");
    if(!fU) {
        spt_CheckError(SPTERR_OS_ERROR, "fopen", NULL);
    }
    result = spt_LoadMatrixTranspose(&U, fU);
    spt_CheckError(result, "main", NULL);
    fclose(fU);

    sscanf(argv[3], "%"PARTI_SCN_INDEX, &mode);
    sscanf(argv[4], "%d", &cuda_dev_id);

    printf("Preheating...\n");
    fflush(stdout);
    int i;
    for(i = 0; i < 2; i++) {
        result = do_ttm(&X, &U, mode, cuda_dev_id);
        spt_CheckError(result, "main", NULL);
    }

    printf("Calculating...\n");
    fflush(stdout);
    for(i = 0; i < 5; i++) {
        result = do_ttm(&X, &U, mode, cuda_dev_id);
        spt_CheckError(result, "main", NULL);
    }

    return 0;
}

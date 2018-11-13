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

#include <ParTI.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/error/error.h"

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

int main() {
    {
        static char bufX[] = "3\n"
            "2 2 2\n"
            "0 0 0 1\n"
            "0 0 1 2\n"
            "0 1 0 3\n"
            "0 1 1 4\n"
            "1 0 0 5\n"
            "1 0 1 6\n"
            "1 1 0 7\n"
            "1 1 1 8\n";
        static char bufU[] = "2 2 2\n"
            "1 2\n"
            "3 4\n";
        FILE *stream = fmemopen(bufX, sizeof bufX - 1, "r");
        sptSparseTensor X;
        int result = sptLoadSparseTensor(&X, 0, stream);
        spt_CheckError(result, "load", NULL);
        fclose(stream);

        stream = fmemopen(bufU, sizeof bufU - 1, "r");
        sptMatrix U;
        result = spt_LoadMatrixTranspose(&U, stream);
        spt_CheckError(result, "load", NULL);
        fclose(stream);

        sptSemiSparseTensor Y;
        result = sptCudaSparseTensorMulMatrix(&Y, &X, &U, 0);
        spt_CheckError(result, "ttm", NULL);

        sptSparseTensor spY;
        result = sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-6);
        spt_CheckError(result, "convert", NULL);

        char *bufY = calloc(1, 1024);
        stream = fmemopen(bufY, 1023, "w");
        result = sptDumpSparseTensor(&spY, 0, stream);
        spt_CheckError(result, "convert", NULL);
        fclose(stream);

        if(strcmp(bufY, "3\n"
                        "2 2 2\n"
                        "0\t0\t0\t11.000000\n"
                        "0\t0\t1\t14.000000\n"
                        "0\t1\t0\t17.000000\n"
                        "0\t1\t1\t20.000000\n"
                        "1\t0\t0\t23.000000\n"
                        "1\t0\t1\t30.000000\n"
                        "1\t1\t0\t37.000000\n"
                        "1\t1\t1\t44.000000\n") != 0) {
            printf("Output mismatch:\n%s", bufY);
            return 1;
        }
        free(bufY);

        sptFreeSparseTensor(&spY);
        sptFreeSemiSparseTensor(&Y);
        sptFreeMatrix(&U);
        sptFreeSparseTensor(&X);
    }
    return 0;
}

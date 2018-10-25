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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <ParTI.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fU, *fY;
    sptSparseTensor X, spU, spY;
    sptSemiSparseTensor Y;
    sptMatrix U;
    size_t mode = 0;
    int cuda_dev_id = -2;

    if(argc < 5) {
        printf("Usage: %s X U Y mode [cuda_dev_id]\n\n", argv[0]);
        exit(1);
    }

    fX = fopen(argv[1], "r");
    if(!fX) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[1]);
        exit(1);
    }
    if(sptLoadSparseTensor(&X, 1, fX)) {
        fprintf(stderr, "Error: failed to load tensor X\n");
        exit(1);
    }
    fclose(fX);

    fU = fopen(argv[2], "r");
    if(!fU) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[2]);
        exit(1);
    }
    if(sptLoadSparseTensor(&spU, 1, fU)) {
        fprintf(stderr, "Error: failed to load tensor U\n");
        exit(1);
    }
    fclose(fU);

    sscanf(argv[4], "%zu", &mode);
    if(argc >= 6) {
        sscanf(argv[5], "%d", &cuda_dev_id);
    }

    if(sptSparseTensorToMatrix(&U, &spU)) {
        fprintf(stderr, "Error: failed to convert U to matrix\n");
        exit(1);
    }
    sptFreeSparseTensor(&spU);

    int result;
    if(cuda_dev_id == -2) {
        result = sptSparseTensorMulMatrix(&Y, &X, &U, mode);
    } else if(cuda_dev_id == -1) {
        result = sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode);
    } else {
        result = sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode);
    }
    if(result) {
        fprintf(stderr, "Error: failed to calculate X*U\n");
        exit(1);
    }

    if(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9)) {
        fprintf(stderr, "Error: failed to convert Y to sparse tensor\n");
        exit(1);
    }

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    fY = fopen(argv[3], "w");
    if(!fY) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[3]);
        exit(1);
    }
    if(sptDumpSparseTensor(&spY, 1, fY)) {
        fprintf(stderr, "Error: failed to dump tensor Y\n");
        exit(1);
    }
    fclose(fY);

    sptFreeSparseTensor(&spY);

    return 0;
}

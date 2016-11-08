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

int main(int argc, char const *argv[]) {
    FILE *fA, *fB, *fY;
    sptSparseTensor A, B, Y;
    int cuda_dev_id = -2;

    if(argc < 4) {
        printf("Usage: %s A B Y [cuda_dev_id]\n\n", argv[0]);
        exit(1);
    }

    fA = fopen(argv[1], "r");
    if(!fA) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[1]);
        exit(1);
    }
    if(sptLoadSparseTensor(&A, 1, fA)) {
        fprintf(stderr, "Error: failed to load tensor A\n");
        exit(1);
    }
    fclose(fA);

    fB = fopen(argv[2], "r");
    if(!fB) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[2]);
        exit(1);
    }
    if(sptLoadSparseTensor(&B, 1, fB)) {
        fprintf(stderr, "Error: failed to load tensor B\n");
        exit(1);
    }
    fclose(fB);

    if(argc >= 6) {
        sscanf(argv[5], "%d", &cuda_dev_id);
    }

    int result;
    if(cuda_dev_id == -2) {
        result = sptSparseTensorDotMulEq(&Y, &A, &B);
    } else if(cuda_dev_id == -1) {
        result = sptOmpSparseTensorDotMulEq(&Y, &A, &B);
    } else {
        result = sptCudaSparseTensorDotMulEq(&Y, &A, &B);
    }
    if(result) {
        fprintf(stderr, "Error: failed to calculate A.*B\n");
        exit(1);
    }

    sptFreeSparseTensor(&B);
    sptFreeSparseTensor(&A);

    fY = fopen(argv[3], "w");
    if(!fY) {
        fprintf(stderr, "Error: failed to open file \"%s\"\n", argv[3]);
        exit(1);
    }
    if(sptDumpSparseTensor(&Y, 1, fY)) {
        fprintf(stderr, "Error: failed to dump tensor Y\n");
        exit(1);
    }
    fclose(fY);

    sptFreeSparseTensor(&Y);

    return 0;
}

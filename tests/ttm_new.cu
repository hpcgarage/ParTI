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
    FILE *fX, *fY;
    sptSparseTensor X, spY;
    sptSemiSparseTensor Y;
    sptMatrix U;
    size_t mode = 0;
    size_t R = 16;
    int cuda_dev_id = -2;
    int niters = 5;

    if(argc < 3) {
        printf("Usage: %s X mode [cuda_dev_id, R, Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    assert(fX != NULL);
    assert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    sscanf(argv[2], "%zu", &mode);
    if(argc >= 4) {
        sscanf(argv[3], "%d", &cuda_dev_id);
    }
    if(argc >= 5) {
        sscanf(argv[4], "%zu", &R);
    }

    fprintf(stderr, "sptRandomizeMatrix(&U, %zu, %zu)\n", X.ndims[mode], R);
    assert(sptRandomizeMatrix(&U, X.ndims[mode], R) == 0);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        assert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(cuda_dev_id == -1) {
        assert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        assert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }

    for(int it=0; it<niters; ++it) {
        sptFreeSemiSparseTensor(&Y);
        if(cuda_dev_id == -2) {
            assert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else if(cuda_dev_id == -1) {
            assert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else {
            sptCudaSetDevice(cuda_dev_id);
            assert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        }
    }

    assert(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    if(argc >= 6) {
        fY = fopen(argv[5], "w");
        assert(fY != NULL);
        assert(sptDumpSparseTensor(&spY, 1, fY) == 0);
        fclose(fY);
    }

    sptFreeSparseTensor(&spY);

    return 0;
}

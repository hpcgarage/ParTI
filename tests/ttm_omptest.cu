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
#include <omp.h>

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
        sptCudaSetDevice(cuda_dev_id);
    }
    if(argc >= 5) {
        sscanf(argv[4], "%zu", &R);
    }

    printf("Tensor: %s, TTM mode %zu\n", argv[1], mode);

    assert(sptRandomizeMatrix(&U, X.ndims[mode], R) == 0);

    static const size_t tmp_ndims[2] = {0, 0};
    sptNewSemiSparseTensor(&Y, 2, 1, tmp_ndims);

    /* We have niters+1 iterations, the first is warm-up */
    for(int nth = 2; nth <= 8; nth += 2) {
        omp_set_num_threads(nth);
        fprintf(stderr, "OMP nthreads=%d\n", nth);
        for(int it=0; it<niters+1; ++it) {
            sptFreeSemiSparseTensor(&Y);
            assert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        }
    }
    fprintf(stderr, "X.nnz = %zu, X.ndims = [%zu, %zu, %zu], Y.nnz = %zu, Y.ndims = [%zu, %zu, %zu]\n", X.nnz, X.ndims[0], X.ndims[1], X.ndims[2], Y.nnz, Y.ndims[0], Y.ndims[1], Y.ndims[2]);

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

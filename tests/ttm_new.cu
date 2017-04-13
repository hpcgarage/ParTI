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

#include <stdio.h>
#include <stdlib.h>
#include <ParTI.h>

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
        printf("Usage: %s X mode impl_num [cuda_dev_id, R, Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    sscanf(argv[2], "%zu", &mode);
    size_t impl_num = 0;
    sscanf(argv[3], "%zu", &impl_num);

    if(argc > 4) {
        sscanf(argv[4], "%d", &cuda_dev_id);
    }
    if(argc > 5) {
        sscanf(argv[5], "%zu", &R);
    }

    fprintf(stderr, "sptRandomizeMatrix(&U, %zu, %zu)\n", X.ndims[mode], R);
    // sptAssert(sptRandomizeMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptNewMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptConstantMatrix(&U, 1) == 0);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(cuda_dev_id == -1) {
        sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        // sptAssert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        abort(); // (sam) FIXME: The following line is commented to pass linking. Uncomment it if it's ready.
        //sptAssert(sptCudaSparseTensorMulMatrixOneKernel(&Y, &X, &U, mode, impl_num) == 0);
    }

    // for(int it=0; it<niters; ++it) {
    //     sptFreeSemiSparseTensor(&Y);
    //     if(cuda_dev_id == -2) {
    //         sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    //     } else if(cuda_dev_id == -1) {
    //         sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    //     } else {
    //         sptCudaSetDevice(cuda_dev_id);
    //         sptAssert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    //     }
    // }

    sptAssert(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    if(argc > 6) {
        fY = fopen(argv[6], "w");
        sptAssert(fY != NULL);
        sptAssert(sptDumpSparseTensor(&spY, 0, fY) == 0);
        fclose(fY);
    }

    sptFreeSparseTensor(&spY);

    return 0;
}

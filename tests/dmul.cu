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

int main(int argc, char *argv[]) {
    FILE *fX, *fY, *fo;
    sptSparseTensor X, Y, Z;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;

    if(argc < 3) {
        printf("Usage: %s X Y [cuda_dev_id, niters, out]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    fY = fopen(argv[1], "r");
    sptAssert(fY != NULL);
    sptAssert(sptLoadSparseTensor(&Y, 1, fY) == 0);
    fclose(fY);

    if(argc >= 4) {
        sscanf(argv[3], "%d", &cuda_dev_id);
    }
    if(argc >= 5) {
        sscanf(argv[4], "%d", &niters);
    }
    sptAssert(niters >= 1);

    // sptSparseTensorSortIndex(&a);
    // sptSparseTensorSortIndex(&b);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        sptAssert(sptSparseTensorDotMulEq(&Z, &X, &Y) == 0);
    } else if(cuda_dev_id == -1) {
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorDotMulEq(&Z, &X, &Y) == 0);
    } else if(cuda_dev_id >= 0) {
        sptCudaSetDevice(cuda_dev_id);
        sptAssert(sptCudaSparseTensorDotMulEq(&Z, &X, &Y) == 0);
    }

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            sptAssert(sptSparseTensorDotMulEq(&Z, &X, &Y) == 0);
        } else if(cuda_dev_id == -1) {
            #pragma omp parallel
            {
                nthreads = omp_get_num_threads();
            }
            printf("nthreads: %d\n", nthreads);
            sptAssert(sptOmpSparseTensorDotMulEq(&Z, &X, &Y) == 0);
        } else if(cuda_dev_id >= 0) {
            sptCudaSetDevice(cuda_dev_id);
            sptAssert(sptCudaSparseTensorDotMulEq(&Z, &X, &Y) == 0);
        }
    }

    if(argc >= 6) {
        fo = fopen(argv[5], "w");
        sptAssert(fo != NULL);
        sptAssert(sptDumpSparseTensor(&Z, 1, fo) == 0);
        fclose(fo);
    }

    sptFreeSparseTensor(&X);
    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&Z);

    return 0;
}

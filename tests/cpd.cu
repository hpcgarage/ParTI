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
#include <omp.h>
#include <ParTI.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fo;
    sptSparseTensor X;
    size_t R = 8;
    int niters = 50;
    double tol = 1e-4;
    sptKruskalTensor ktensor;
    int nloops = 5;
    int cuda_dev_id = -2;
    int nthreads;
    int use_reduce = 1;

    if(argc < 2) {
        printf("Usage: %s X [cuda_dev_id, nthreads, R, use_reduce, ktensor]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    sptSparseTensorStatus(&X, stdout);
    // sptDumpSparseTensor(&X, 0, stdout);

    if(argc >= 3) {
        sscanf(argv[2], "%d", &cuda_dev_id);
    }
    if(argc >= 4) {
        sscanf(argv[3], "%d", &nthreads);
    }
    if(argc >= 5) {
        sscanf(argv[4], "%zu", &R);
    }
    if(argc >= 6) {
        sscanf(argv[5], "%d", &use_reduce);
    }

    size_t nmodes = X.nmodes;
    sptNewKruskalTensor(&ktensor, nmodes, X.ndims, R);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        sptAssert(sptCpdAls(&X, R, niters, tol, &ktensor) == 0);
    } else if(cuda_dev_id == -1) {
        omp_set_num_threads(nthreads);
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        printf("use_reduce: %d\n", use_reduce);
        sptAssert(sptOmpCpdAls(&X, R, niters, tol, nthreads, use_reduce, &ktensor) == 0);
    } /* else {
         sptCudaSetDevice(cuda_dev_id);
         sptAssert(sptCudaCpdAls(&X, R, niters, tol, &ktensor) == 0);
    } */

    // for(int it=0; it<nloops; ++it) {
    //     if(cuda_dev_id == -2) {
    //         nthreads = 1;
    //         sptAssert(sptCpdAls(&X, R, niters, tol, &ktensor) == 0);
    //     } else if(cuda_dev_id == -1) {
    //         #pragma omp parallel
    //         {
    //             nthreads = omp_get_num_threads();
    //         }
    //         printf("nthreads: %d\n", nthreads);
    //         sptAssert(sptOmpCpdAls(&X, R, niters, tol, &ktensor) == 0);
    //     } else {
    //          sptCudaSetDevice(cuda_dev_id);
    //          // sptAssert(sptCudaCpdAls(&X, R, niters, tol, &ktensor) == 0);
    //     }
    // }

    sptFreeSparseTensor(&X);
    sptFreeKruskalTensor(&ktensor);

    if(argc >= 7) {
        // Dump ktensor to files
        fo = fopen(argv[6], "w");
        sptAssert( sptDumpKruskalTensor(&ktensor, 0, fo) == 0 );
        fclose(fo);
    }

    return 0;
}

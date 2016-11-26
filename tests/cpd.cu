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
#include <omp.h>
#include <SpTOL.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fo;
    sptSparseTensor X;
    size_t R = 16;
    int niters = 1;
    double tol = 1e-4;
    sptKruskalTensor ktensor;
    int cuda_dev_id = -2;
    int nthreads;

    if(argc < 2) {
        printf("Usage: %s X [cuda_dev_id, R, ktensor]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    assert(fX != NULL);
    assert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    sptDumpSparseTensor(&X, 0, stdout);

    if(argc >= 3) {
        sscanf(argv[2], "%d", &cuda_dev_id);
    }
    if(argc >= 4) {
        sscanf(argv[3], "%zu", &R);
    }

    size_t nmodes = X.nmodes;

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        assert(sptCpdAls(&X, R, niters, tol, &ktensor) == 0);
    } 


    sptFreeSparseTensor(&X);
    // sptFreeKruskalTensor(&ktensor);

    if(argc >= 5) {
        // Dump ktensor to files
    }

    return 0;
}

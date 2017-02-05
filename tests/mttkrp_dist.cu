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
#include "../src/sptensor/sptensor.h"

int main(int argc, char const *argv[]) {
    FILE *fX, *fo;
    sptSparseTensor X;
    sptMatrix ** U;
    sptSizeVector mats_order;
    sptVector scratch;
    size_t mode = 0;
    size_t R = 16;
    int niters = 5;

    if(argc < 3) {
        printf("Usage: %s X mode split_size batch_size cuda_dev_ids... [R Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    printf("input file: %s\n", argv[1]); fflush(stdout);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    size_t nmodes = X.nmodes;

    sscanf(argv[2], "%zu", &mode);
    size_t *split_size = new size_t[nmodes];
    for(size_t i = 0; i < nmodes; ++i) {
        sscanf(argv[i+3], "%zu", &split_size[i]);
    }
    size_t batch_size;
    sscanf(argv[nmodes+3], "%zu", &batch_size);
    int *gpu_map = new int[batch_size];
    for(size_t i = 0; i < batch_size; ++i) {
        sscanf(argv[nmodes+i+4], "%d", &gpu_map[i]);
    }

    if(argc > nmodes+batch_size+4u) {
        sscanf(argv[nmodes+batch_size+4], "%zu", &R);
    }

    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(size_t m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    size_t max_ndims = 0;
    for(size_t m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    size_t stride = U[0]->stride;


    sptNewSizeVector(&mats_order, nmodes-1, nmodes-1);
    size_t j = 0;
    for(int m=nmodes-1; m>=0; --m) {
        if(m != (int)mode) {
            mats_order.data[j] = m;
            ++ j;
        }
    }

    spt_SplitResult *splits;
    size_t nsplits;

    sptAssert(spt_SparseTensorGetAllSplits(
        &splits,
        &nsplits,
        &X,
        split_size,
        NULL,
        1
    ) == 0);
    sptAssert(sptCudaDistributedMTTKRP(
        splits,
        nsplits,
        batch_size,
        U,
        mats_order.data,
        mode,
        gpu_map
    ) == 0);

    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&X);
    sptFreeSizeVector(&mats_order);

    if(argc > nmodes+batch_size+4u) {
        fo = fopen(argv[nmodes+batch_size+4], "w");
        sptAssert(fo != NULL);
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);

    delete[] gpu_map;
    delete[] split_size;

    return 0;
}

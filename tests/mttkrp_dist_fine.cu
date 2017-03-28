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

#define FINEGRAIN

template <typename T>
static void print_array(const T array[], size_t length, T start_index) {
    if(length == 0) {
        return;
    }
    printf("%d", (int) (array[0] + start_index));
    size_t i;
    for(i = 1; i < length; ++i) {
        printf(", %d", (int) (array[i] + start_index));
    }
}

int main(int argc, char const *argv[]) 
{
    /* 1: Coarse grain; 2: fine grain; 3: medium grain */
    int split_grain = 2;
    FILE *fX, *fo;
    sptSparseTensor tsr;
    sptMatrix ** U;
    size_t mode = 0;
    size_t R = 16;

    if(argc < 7) {
        printf("Usage: %s tsr mode mem_size nbatchs batch_size cuda_dev_ids... [R Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    printf("input file: %s\n", argv[1]); fflush(stdout);
    sptAssert(sptLoadSparseTensor(&tsr, 1, fX) == 0);
    fclose(fX);

    size_t const nmodes = tsr.nmodes;
    size_t * const ndims = tsr.ndims;

    sscanf(argv[2], "%zu", &mode);
    printf("Mode = %zu\n", mode);

    size_t mem_size;
    sscanf(argv[3], "%zu", &mem_size);
    printf("mem_size = %zu\n", mem_size);

    size_t wordsize = (sizeof(size_t) > sizeof(sptScalar)) ? sizeof(size_t) : sizeof(sptScalar);
    size_t memwords = mem_size / wordsize;
    printf("memwords: %zu\n", memwords);

    size_t nbatchs;
    sscanf(argv[4], "%zu", &nbatchs);
    printf("nbatchs = %zu\n", nbatchs);

    size_t batch_size;
    sscanf(argv[5], "%zu", &batch_size);
    printf("Batch_size = %zu\n", batch_size);

    int *gpu_map = new int[batch_size];
    for(size_t i = 0; i < batch_size; ++i) {
        sscanf(argv[i+6], "%d", &gpu_map[i]);
    }
    printf("Gpu_map = [");
    print_array(gpu_map, batch_size, 0);
    printf("]\n");

    if((unsigned) argc > batch_size+6) {
        sscanf(argv[batch_size+6], "%zu", &R);
    }
    printf("R = %zu\n", R);
    printf("Tensor NNZ: %zu\n", tsr.nnz);

    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(size_t m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    size_t max_ndims = 0;
    for(size_t m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], tsr.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], tsr.ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(tsr.ndims[m] > max_ndims)
        max_ndims = tsr.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);

    size_t * mats_order = (size_t*)malloc(nmodes * sizeof(size_t));
    mats_order[0] = mode;
    for(size_t i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;
    printf("mats_order:\n");
    spt_DumpArray(mats_order, nmodes, 0, stdout);

    size_t queue_size = nbatchs * batch_size;
    printf("queue_size: %zu (%zu * %zu)\n", queue_size, nbatchs, batch_size);

    size_t nsplits = 0;    // Total 
    size_t split_nnz_len = 0;
    size_t nnz_split_begin = 0;
    size_t nnz_split_next = 0;
    double queue_time = 0, total_time = 0;


    sptAssert(spt_ComputeFineSplitParameters(&split_nnz_len, &tsr, R, memwords) == 0);
    printf("Calculated split_nnz_len: %zu\n", split_nnz_len);

    spt_SplitResult *splits = (spt_SplitResult *)malloc(queue_size * sizeof(spt_SplitResult));
    size_t splits_idx = 0;
    while (nnz_split_next < tsr.nnz) {
        printf("nnz_split_begin: %zu, nnz_split_next: %zu\n", nnz_split_begin, nnz_split_next);
        nnz_split_begin = nnz_split_next;

        size_t real_queue_size = 0;
        sptAssert(spt_FineSplitSparseTensorBatch(
            splits,
            &nnz_split_next,
            &real_queue_size,
            queue_size,
            split_nnz_len,
            &tsr,
            nnz_split_begin
        ) == 0);
        printf("real_queue_size: %zu\n", real_queue_size);
        nsplits += real_queue_size;
        // spt_SparseTensorDumpAllSplits(splits, queue_size, stdout);

        sptAssert(sptCudaDistributedMTTKRP(
            &queue_time,
            split_grain,
            splits,
            real_queue_size,
            batch_size,
            U,
            mats_order,
            mode,
            gpu_map
        ) == 0);
        total_time += queue_time;

        spt_SparseTensorFreeAllSplits(splits, real_queue_size);

        ++ splits_idx;
    }   // split the whole tensor
    free(splits);

    printf("\n[CUDA SpTns Fine-Dist MTTKRP]: %lf s\n\n", total_time);  

    

    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&tsr);
    free(mats_order);

    if((unsigned) argc > batch_size+7) {
        printf("Output = %s\n", argv[batch_size+7]);
        fo = fopen(argv[batch_size+7], "w");
        sptAssert(fo != NULL);
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);

    delete[] gpu_map;

    return 0;
}

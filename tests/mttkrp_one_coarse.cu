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

#define COARSEGRAIN

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
    int split_grain = 1;
    FILE *fX, *fo;
    sptSparseTensor tsr;
    sptMatrix ** U;
    size_t mode = 0;
    size_t R = 4;
    size_t max_nstreams = 4;

    if(argc < 3) {
        printf("Usage: %s tsr mode smem_size nstreams nblocks [R Y max_nstreams]\n\n", argv[0]);
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

    size_t smem_size;
    sscanf(argv[3], "%zu", &smem_size);
    printf("smem_size = %zu\n", smem_size);

    size_t wordsize = (sizeof(size_t) > sizeof(sptScalar)) ? sizeof(size_t) : sizeof(sptScalar);
    size_t smemwords = smem_size / wordsize;
    printf("smemwords: %zu\n", smemwords);

    size_t nstreams;
    sscanf(argv[4], "%zu", &nstreams);
    printf("nstreams = %zu\n", nstreams);

    size_t nblocks;
    sscanf(argv[5], "%zu", &nblocks);
    printf("nblocks = %zu\n", nblocks);


    if((unsigned) argc > 6) {
        sscanf(argv[6], "%zu", &R);
    }
    printf("R = %zu\n", R);
    printf("Tensor NNZ: %zu\n", tsr.nnz);

    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(size_t m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    size_t max_ndims = 0;
    for(size_t m=0; m<nmodes; ++m) {
      // sptAssert(sptRandomizeMatrix(U[m], ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
      if(ndims[m] > max_ndims)
        max_ndims = ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);


    size_t * mats_order = (size_t*)malloc(nmodes * sizeof(size_t));
    mats_order[0] = mode;
    for(size_t i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;
    printf("mats_order:\n");
    spt_DumpArray(mats_order, nmodes, 0, stdout);

    sptSparseTensorSortIndexCustomOrder(&tsr, mats_order, 1);  // tsr sorted from mode-0, ..., N-1.
    // sptDumpSparseTensor(&tsr, 0, stdout);

    size_t * slice_nnzs = (size_t *)malloc(ndims[mode] * sizeof(size_t));
    sptAssert( spt_ComputeSliceSizes(slice_nnzs, &tsr, mode) == 0 );
    printf("slice_nnzs: \n");
    spt_DumpArray(slice_nnzs, ndims[mode], 0, stdout);

    size_t queue_size = nstreams * nblocks;
    printf("queue_size: %zu (%zu * %zu)\n", queue_size, nstreams, nblocks);
    
    size_t nsplits = 0;    // Total 
    size_t * split_idx_len = (size_t*)malloc(queue_size * sizeof(size_t));
    size_t nnz_split_begin = 0;
    size_t nnz_split_next = 0;
    double queue_time = 0, total_time = 0;

    spt_SplitResult *splits = (spt_SplitResult *)malloc(queue_size * sizeof(spt_SplitResult));
    while (nnz_split_next < tsr.nnz) {
        printf("nnz_split_begin: %zu, nnz_split_next: %zu\n", nnz_split_begin, nnz_split_next);
        nnz_split_begin = nnz_split_next;
        size_t idx_begin = tsr.inds[mode].data[nnz_split_begin];
        sptAssert(spt_ComputeCoarseSplitParameters(split_idx_len, queue_size, &tsr, slice_nnzs, idx_begin, mode, R, smemwords) == 0);
        // printf("idx_begin: %zu\n", idx_begin);
        printf("Calculated split_idx_len: \n");
        spt_DumpArray(split_idx_len, queue_size, 0, stdout);

        size_t real_queue_size = 0;
        sptAssert(spt_CoarseSplitSparseTensorBatch(
            splits,
            &nnz_split_next,
            &real_queue_size,
            queue_size,
            split_idx_len,
            mode,
            &tsr,
            nnz_split_begin
        ) == 0);
        printf("real_queue_size: %zu\n", real_queue_size);
        nsplits += real_queue_size;
        spt_SparseTensorDumpAllSplits(splits, queue_size, stdout);
 
        sptAssert(sptCudaOneMTTKRP(
            &queue_time,
            split_grain,
            &tsr,
            splits,
            real_queue_size,
            nblocks,
            U,
            mats_order,
            mode, 
            max_nstreams
        ) == 0);
        total_time += queue_time;

        spt_SparseTensorFreeAllSplits(splits, real_queue_size);
        
    }   // Split the whole tensor  
    free(splits);

    printf("Total nsplits: %zu\n", nsplits);
    printf("\n[CUDA SpTns Coarse-One MTTKRP]: %lf s\n\n", total_time);  
    


    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&tsr);
    free(mats_order);

    if((unsigned) argc > 7) {
        printf("Output = %s\n", argv[7]);
        fo = fopen(argv[7], "w");
        sptAssert(fo != NULL);
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);
    free(split_idx_len);
    free(slice_nnzs);
    

    return 0;
}

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


int main(int argc, char const *argv[]) 
{
    /* 1: Coarse grain; 2: fine grain; 3: medium grain */
    int split_grain = 2;
    FILE *fX, *fo;
    sptSparseTensor tsr;
    sptMatrix ** U;
    size_t mode = 0;
    size_t R = 16;
    size_t max_nstreams = 4;
    size_t const max_nthreads_per_block = 512;
    size_t const max_nthreadsy = 16;
    size_t max_nthreadsx = 256;
    int arg_loc = 0;

    if(argc < 8) {
        printf("Usage: %s tsr mode impl_num smem_size nstreams nblocks cuda_dev_id [R max_nstreams Y]\n\n", argv[0]);
        return 1;
    }

    ++ arg_loc;
    fX = fopen(argv[arg_loc], "r");
    ++ arg_loc;
    sptAssert(fX != NULL);
    printf("input file: %s\n", argv[1]); fflush(stdout);
    sptAssert(sptLoadSparseTensor(&tsr, 1, fX) == 0);
    fclose(fX);

    size_t const nmodes = tsr.nmodes;
    size_t * const ndims = tsr.ndims;

    sscanf(argv[arg_loc], "%zu", &mode);
    ++ arg_loc;
    printf("Mode = %zu\n", mode);

    size_t impl_num;
    sscanf(argv[arg_loc], "%zu", &impl_num);
    ++ arg_loc;
    printf("impl_num = %zu\n", impl_num);

    size_t smem_size;
    sscanf(argv[arg_loc], "%zu", &smem_size);
    ++ arg_loc;
    printf("smem_size = %zu\n", smem_size);

    size_t wordsize = (sizeof(size_t) > sizeof(sptScalar)) ? sizeof(size_t) : sizeof(sptScalar);
    size_t smemwords = smem_size / wordsize;
    printf("smemwords: %zu\n", smemwords);

    /* nstreams > max_nstreams */
    size_t nstreams;
    sscanf(argv[arg_loc], "%zu", &nstreams);
    ++ arg_loc;
    printf("nstreams = %zu\n", nstreams);

    size_t nblocks;
    sscanf(argv[arg_loc], "%zu", &nblocks);
    ++ arg_loc;
    printf("nblocks = %zu\n", nblocks);

    size_t cuda_dev_id;
    sscanf(argv[arg_loc], "%zu", &cuda_dev_id);
    ++ arg_loc;
    printf("cuda_dev_id = %zu\n", cuda_dev_id);

    if(argc > arg_loc) {
        sscanf(argv[arg_loc], "%zu", &R);
        ++ arg_loc;
    }
    printf("R = %zu\n", R);

    if(argc > arg_loc) {
        sscanf(argv[arg_loc], "%zu", &max_nstreams);
        ++ arg_loc;
    }
    printf("max_nstreams = %zu\n", max_nstreams);

    printf("Tensor NNZ: %zu\n", tsr.nnz);

    if(impl_num != 11)
        max_nthreadsx = (size_t) max_nthreads_per_block / max_nthreadsy;
    printf("max_nthreadsx: %zu\n", max_nthreadsx);

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
    size_t stride = U[nmodes]->stride;
    printf("stride: %zu\n", stride);


    size_t * mats_order = (size_t*)malloc(nmodes * sizeof(size_t));
    mats_order[0] = mode;
    for(size_t i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;
    printf("mats_order:\n");
    spt_DumpArray(mats_order, nmodes, 0, stdout);


    size_t queue_size = nstreams * nblocks;
    printf("queue_size: %zu (%zu * %zu)\n", queue_size, nstreams, nblocks);
    
    size_t nsplits = 0;    // Total 
    size_t split_nnz_len = 0;
    size_t nnz_split_begin = 0;
    size_t nnz_split_next = 0;
    double queue_time = 0, total_time = 0;

    sptAssert(spt_ComputeFineSplitParametersOne(&split_nnz_len, &tsr, max_nthreadsx) == 0);
    printf("Calculated split_nnz_len: %zu\n", split_nnz_len);

    spt_SplitResult *splits = (spt_SplitResult *)malloc(queue_size * sizeof(spt_SplitResult));
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
        sptAssert(real_queue_size <= queue_size);
        nsplits += real_queue_size;
        // spt_SparseTensorDumpAllSplits(splits, queue_size, stdout);
 
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
            nnz_split_begin,
            max_nstreams,
            max_nthreadsy,
            impl_num,
            cuda_dev_id
        ) == 0);
        total_time += queue_time;

        spt_SparseTensorFreeAllSplits(splits, real_queue_size);
        
    }   // Split the whole tensor  
    free(splits);

    // sptDumpMatrix(U[nmodes], stdout);

    printf("Total nsplits: %zu\n", nsplits);
    printf("\n[CUDA SpTns Fine-One MTTKRP]: %lf s\n\n", total_time);  
    


    for(size_t m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    sptFreeSparseTensor(&tsr);
    free(mats_order);

    if(argc > arg_loc) {
        printf("Output = %s\n", argv[arg_loc]);
        fo = fopen(argv[arg_loc], "w");
        ++ arg_loc;
        sptAssert(fo != NULL);
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }

    sptFreeMatrix(U[nmodes]);
    free(U);
    

    return 0;
}

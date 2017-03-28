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

#ifndef PARTI_SPTENSOR_H
#define PARTI_SPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ParTI.h>
#include "../error/error.h"

int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2);
int spt_SparseTensorCompareIndicesRange(const sptSparseTensor *tsr, size_t loc, size_t * const inds1, size_t * const inds2);
void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2);

void spt_SparseTensorCollectZeros(sptSparseTensor *tsr);

int spt_DistSparseTensor(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);

int spt_DistSparseTensorFixed(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);

int spt_GetSubSparseTensor(sptSparseTensor *dest, const sptSparseTensor *tsr, const size_t limit_low[], const size_t limit_high[]);

struct spt_TagSplitHandle {
    size_t nsplits;
    sptSparseTensor *tsr;
    size_t *max_size_by_mode;
    size_t *inds_low;
    size_t *inds_high;
    size_t level;
    int *resume_branch;
    size_t *cut_idx;
    size_t *cut_low;
};
typedef struct spt_TagSplitHandle *spt_SplitHandle;
int spt_StartSplitSparseTensor(spt_SplitHandle *handle, const sptSparseTensor *tsr, const size_t max_size_by_mode[]);
int spt_SplitSparseTensor(sptSparseTensor *dest, size_t *inds_low, size_t *inds_high, spt_SplitHandle handle);
void spt_FinishSplitSparseTensor(spt_SplitHandle handle);

typedef struct spt_TagSplitResult {
    sptSparseTensor tensor;
    size_t *inds_low;
    size_t *inds_high;
    struct spt_TagSplitResult *next;    // Not use now, for one gpu implementation. Now all splits inside one queue has a real subtsr contigously, the length is marked by real_queue_size.
} spt_SplitResult;
/* FIXME: index_limit_by_mode is not used yet */
int spt_SparseTensorGetAllSplits(spt_SplitResult **splits, size_t *nsplits, const sptSparseTensor *tsr, const size_t nnz_limit_by_mode[], const size_t index_limit_by_mode[], int emit_map);
// void spt_SparseTensorFreeAllSplits(spt_SplitResult *splits);
void spt_SparseTensorFreeAllSplits(spt_SplitResult *splits, size_t const nsplits);
int spt_SparseTensorDumpAllSplits(const spt_SplitResult * splits, size_t const nsplits, FILE *fp);
void spt_DumpArray(const size_t array[], size_t length, size_t start_index, FILE *fp);

// abundant
int spt_SparseTensorBalancedSplit(
    spt_SplitResult **splits, 
    size_t *nsplits, 
    sptSparseTensor *tsr, 
    const size_t nnz_limit,
    const size_t index_limit_by_mode[]);

/* Coarse-grain split */
int spt_ComputeCoarseSplitParameters(
    size_t * split_idx_len,
    size_t const nsplits,
    sptSparseTensor * const tsr,
    size_t * const slice_nnzs,
    size_t const idx_begin,
    size_t const mode,
    size_t const stride,
    size_t const memwords);

int spt_ComputeCoarseSplitParametersOne(
    size_t * split_idx_len, // size: nsplits
    size_t const nsplits,
    sptSparseTensor * const tsr,
    size_t * const slice_nnzs,
    size_t const idx_begin,
    size_t const mode,
    size_t const stride,
    size_t const memwords,
    size_t const max_nthreads_per_block);
    
int spt_CoarseSplitSparseTensorBatch(
    spt_SplitResult * splits,
    size_t * nnz_split_next,
    size_t * real_nsplits,
    size_t const nsplits,
    size_t * const split_idx_len,
    const size_t mode,
    sptSparseTensor * tsr,
    size_t const nnz_split_begin);

// abundant
int spt_CoarseSplitSparseTensorAll(
    spt_SplitResult ** splits,
    size_t * nsplits,
    const size_t split_idx_len,
    const size_t mode,
    sptSparseTensor * tsr);

// abundant
int sptCoarseSplitSparseTensor(sptSparseTensor *tsr, const int num, sptSparseTensor *cstsr);

int spt_CoarseSplitSparseTensorStep(
    spt_SplitResult * splits,
    size_t * nnz_ptr_next,
    const size_t split_idx_len,
    const size_t mode,
    const sptSparseTensor * tsr,
    const size_t nnz_ptr_begin);


/* Fine-grain split */
int spt_ComputeFineSplitParameters(
    size_t * split_nnz_len, // Scalar
    sptSparseTensor * const tsr,
    size_t const stride,
    size_t const memwords);
    
int spt_FineSplitSparseTensorBatch(
    spt_SplitResult * splits,
    size_t * nnz_split_next,
    size_t * real_nsplits,
    const size_t nsplits,
    const size_t split_nnz_len,
    sptSparseTensor * tsr,
    size_t const nnz_split_begin);

int spt_FineSplitSparseTensorStep(
    spt_SplitResult * split,
    size_t * nnz_ptr_next,
    const size_t split_nnz_len,
    sptSparseTensor * tsr,
    const size_t nnz_ptr_begin);


/* Medium-grain split */
int spt_ComputeMediumSplitParameters(
    size_t * split_idx_len, // size: nmodes
    sptSparseTensor * const tsr,
    size_t const stride,
    size_t const memwords);

int spt_MediumSplitSparseTensorBatch(
    spt_SplitResult * splits,
    size_t * nnz_split_next,
    size_t * real_nsplits,
    size_t const nsplits,
    size_t * const split_idx_lens,
    sptSparseTensor * tsr,
    size_t const nnz_split_begin,
    size_t * est_inds_low,
    size_t * est_inds_high);
    
int spt_MediumSplitSparseTensorStep(    // In-place
    spt_SplitResult * split,
    size_t * nnz_ptr_next,
    size_t * subnnz,
    size_t * const split_idx_lens,
    sptSparseTensor * tsr,
    const size_t nnz_ptr_begin,
    size_t * const est_inds_low,
    size_t * const est_inds_high);



#ifdef __cplusplus
}
#endif

#endif

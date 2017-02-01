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

void spt_SparseTensorCollectZeros(sptSparseTensor *tsr);

int spt_DistSparseTensor(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);

int spt_DistSparseTensorFixed(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);

int spt_SliceSparseTensor(sptSparseTensor *dest, const sptSparseTensor *tsr, const size_t limit_low[], const size_t limit_high[]);

typedef struct spt_TagSplitHandle *spt_SplitHandle;
int spt_StartSplitSparseTensor(spt_SplitHandle *handle, const sptSparseTensor *tsr, const size_t max_size_by_mode[]);
int spt_SplitSparseTensor(sptSparseTensor *dest, size_t *inds_low, size_t *inds_high, spt_SplitHandle handle);
void spt_FinishSplitSparseTensor(spt_SplitHandle handle);

typedef struct spt_TagSplitResult {
    sptSparseTensor tensor;
    size_t *inds_low;
    size_t *inds_high;
    struct spt_TagSplitResult *next;
} spt_SplitResult;
/* FIXME: index_limit_by_mode is not used yet */
int spt_SparseTensorGetAllSplits(spt_SplitResult **splits, size_t *nsplits, const sptSparseTensor *tsr, const size_t nnz_limit_by_mode[], const size_t index_limit_by_mode[], int emit_map);
void spt_SparseTensorFreeAllSplits(spt_SplitResult *splits);

#ifdef __cplusplus
}
#endif

#endif

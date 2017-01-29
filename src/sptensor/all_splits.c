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

#include <assert.h>
#include <stdlib.h>
#include <ParTI.h>
#include "sptensor.h"
#include "../error/error.h"

static size_t spt_CalcMaxSplitCount(const sptSparseTensor *tsr, const size_t cuts_by_mode[]) {
    size_t prod = 1;
    size_t m;
    for(m = 0; m < tsr->nmodes; ++m) {
        prod *= cuts_by_mode[m];
    }
    return prod+1;
}

static int spt_FindSplitStep(const sptSparseTensor *tsr, size_t mode, size_t cut_point) {
    if(cut_point == 0) {
        if(tsr->nnz == 0) {
            return 0;
        } else {
            ++cut_point;
        }
    }
    while(cut_point < tsr->nnz &&
        tsr->inds[mode].data[cut_point-1] == tsr->inds[mode].data[cut_point]) {
            ++cut_point;
    }
    return cut_point;
}

static int spt_SparseTensorPartialSplit(spt_SplitResult *splits, size_t *nsplits, sptSparseTensor *tsr, const size_t cuts_by_mode[], int emit_map, size_t level) {
    int result;
    if(level >= tsr->nmodes) {
        splits[*nsplits].tensor = tsr;
        ++*nsplits;
        return 0;
    }
    if(cuts_by_mode[level] == 1) {
        return spt_SparseTensorPartialSplit(splits, nsplits, tsr, cuts_by_mode, emit_map, level);
    }

    sptSparseTensorSortIndexAtMode(tsr, level);
    size_t cut_idx;
    size_t cut_low = 0;
    for(cut_idx = 0; cut_idx < cuts_by_mode[level]; ++cut_idx) {
        size_t cut_high = (tsr->nnz * (cut_idx+1) * 2 + 1) / (cuts_by_mode[level] * 2);
        assert(cut_high <= tsr->nnz);
        cut_high = spt_FindSplitStep(tsr, level, cut_high);
        if(cut_high <= cut_low) {
            continue;
        }

        sptSparseTensor *subtsr = malloc(sizeof *subtsr);
        spt_CheckOSError(subtsr == NULL, "SpTns PartSplt");
        sptNewSparseTensor(subtsr, tsr->nmodes, tsr->ndims);
        size_t m;
        for(m = 0; m < subtsr->nmodes; ++m) {
            sptNewSizeVector(&subtsr->inds[m], cut_high - cut_low, cut_high - cut_low);
            memcpy(subtsr->inds[m].data, &tsr->inds[m].data[cut_low], (cut_high-cut_low) * sizeof (size_t));
        }
        sptNewVector(&subtsr->values, cut_high - cut_low, cut_high - cut_low);
        memcpy(subtsr->values.data, &tsr->values.data[cut_low], (cut_high-cut_low) * sizeof (sptScalar));
        tsr->sortkey = level;
        tsr->nnz = cut_high - cut_low;

        result = spt_SparseTensorPartialSplit(splits, nsplits, subtsr, cuts_by_mode, emit_map, level+1);
        spt_CheckError(result, "SpTns PartSplt", NULL);

        cut_low = cut_high;
    }
    sptFreeSparseTensor(tsr);
    free(tsr);

    return 0;
}

/**
 * A convenient function to get all splits by repeatively calling `spt_SplitSparseTensor`
 *
 * @param[out] splits        Place to store all splits
 * @param[out] nsplits       Place to store the number of actual splits
 * @param[in]  tsr           The tensor to split
 * @param[in]  cuts_by_mode  The number of cuts at each mode, length `tsr->nmodes`
 */
int spt_SparseTensorGetAllSplits(spt_SplitResult **splits, size_t *nsplits, const sptSparseTensor *tsr, const size_t cuts_by_mode[], int emit_map) {
    int result;
    size_t nsplits_bak;
    if(nsplits == NULL) {
        nsplits = &nsplits_bak;
    }
    *splits = calloc(spt_CalcMaxSplitCount(tsr, cuts_by_mode), sizeof (*splits)[0]);
    spt_CheckOSError(*splits == NULL, "SpTns AllSplts");

    sptSparseTensor *tsr_copy = malloc(sizeof *tsr_copy);
    spt_CheckOSError(tsr_copy == NULL, "SpTns AllSplts");
    result = sptCopySparseTensor(tsr_copy, tsr);
    spt_CheckError(result, "SpTns AllSplts", NULL);

    result = spt_SparseTensorPartialSplit(*splits, nsplits, tsr_copy, cuts_by_mode, emit_map, 0);
    spt_CheckError(result, "SpTns AllSplts", NULL);
    return 0;
}

/**
 * Free all sub-tensors created by `spt_SparseTensorGetAllSplits`
 */
void spt_SparseTensorFreeAllSplits(spt_SplitResult splits[], size_t nsplits) {
    size_t i;
    assert(splits[nsplits].tensor == NULL);
    for(i = 0; i < nsplits; ++i) {
        sptFreeSparseTensor(splits[i].tensor);
        free(splits[i].tensor);
    }
    free(splits);
}

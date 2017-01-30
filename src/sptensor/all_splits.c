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

static void spt_RotateMode(sptSparseTensor *tsr) {
    if(tsr->nmodes == 0) {
        return;
    }

    sptSizeVector inds0 = tsr->inds[0];
    memmove(&tsr->inds[0], &tsr->inds[1], (tsr->nmodes-1) * sizeof (sptSizeVector));
    tsr->inds[tsr->nmodes-1] = inds0;
}

static int spt_SparseTensorPartialSplit(spt_SplitResult ***splits_end, size_t *nsplits, sptSparseTensor *tsr, const size_t cuts_by_mode[], int emit_map, size_t level) {
    int result;
    if(level >= tsr->nmodes) {
        **splits_end = malloc(sizeof ***splits_end);
        (**splits_end)->next = NULL;
        (**splits_end)->tensor = *tsr;
        *splits_end = &(**splits_end)->next;
        if(nsplits) {
            ++*nsplits;
        }
        return 0;
    }
    if(cuts_by_mode[level] == 1) {
        spt_RotateMode(tsr);
        sptSparseTensorSortIndex(tsr);
        return spt_SparseTensorPartialSplit(splits_end, nsplits, tsr, cuts_by_mode, emit_map, level+1);
    }

    size_t cut_idx;
    size_t cut_low = 0;
    for(cut_idx = 0; cut_idx < cuts_by_mode[level]; ++cut_idx) {
        size_t cut_high = tsr->nnz * (cut_idx+1) / cuts_by_mode[level];
        assert(cut_high <= tsr->nnz);
        cut_high = spt_FindSplitStep(tsr, level, cut_high);
        if(cut_high <= cut_low) {
            continue;
        }

        sptSparseTensor subtsr;
        result = sptNewSparseTensor(&subtsr, tsr->nmodes, tsr->ndims);
        spt_CheckError(result, "SpTns PartSplt", NULL);
        size_t m;
        for(m = 0; m < subtsr.nmodes; ++m) {
            result = sptNewSizeVector(&subtsr.inds[m], cut_high - cut_low, cut_high - cut_low);
            spt_CheckError(result, "SpTns PartSplt", NULL);
            memcpy(subtsr.inds[m].data, &tsr->inds[m].data[cut_low], (cut_high-cut_low) * sizeof (size_t));
        }
        result = sptNewVector(&subtsr.values, cut_high - cut_low, cut_high - cut_low);
        spt_CheckError(result, "SpTns PartSplt", NULL);
        memcpy(subtsr.values.data, &tsr->values.data[cut_low], (cut_high-cut_low) * sizeof (sptScalar));
        subtsr.nnz = cut_high - cut_low;

        spt_RotateMode(&subtsr);
        sptSparseTensorSortIndex(&subtsr);
        result = spt_SparseTensorPartialSplit(splits_end, nsplits, &subtsr, cuts_by_mode, emit_map, level+1);
        spt_CheckError(result, "SpTns PartSplt", NULL);

        cut_low = cut_high;
    }

    sptFreeSparseTensor(tsr);
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

    spt_SplitResult **splits_end = splits;

    if(nsplits) {
        *nsplits = 0;
    }

    sptSparseTensor tsr_copy;
    result = sptCopySparseTensor(&tsr_copy, tsr);
    spt_CheckError(result, "SpTns AllSplts", NULL);

    result = spt_SparseTensorPartialSplit(&splits_end, nsplits, &tsr_copy, cuts_by_mode, emit_map, 0);
    spt_CheckError(result, "SpTns AllSplts", NULL);

    return 0;
}

/**
 * Free all sub-tensors created by `spt_SparseTensorGetAllSplits`
 */
void spt_SparseTensorFreeAllSplits(spt_SplitResult *splits) {
    while(splits) {
        spt_SplitResult *temp = splits;
        sptFreeSparseTensor(&splits->tensor);
        splits = splits->next;
        free(temp);
    }
}

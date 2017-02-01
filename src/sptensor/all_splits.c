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

static int spt_FindSplitStep(const sptSparseTensor *tsr, size_t cut_point, int direction) {
    if(direction) {
        if(cut_point == 0) {
            ++cut_point;
        }
        while(cut_point < tsr->nnz &&
            tsr->inds[0].data[cut_point-1] == tsr->inds[0].data[cut_point]) {
                ++cut_point;
        }
    } else {
        while(cut_point != 0 &&
            tsr->inds[0].data[cut_point-1] == tsr->inds[0].data[cut_point]) {
                --cut_point;
        }
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

static int spt_SparseTensorPartialSplit(spt_SplitResult ***splits_end, size_t *nsplits, sptSparseTensor *tsr, const size_t nnz_limit_by_mode[], const size_t index_limit_by_mode[], int emit_map, size_t inds_low[], size_t inds_high[], size_t level) {
    int result;

    /* Do nothing with an empty tensor */
    if(tsr->nnz == 0) {
        sptFreeSparseTensor(tsr);
        return 0;
    }

    /* All modes cut, finish the recursion */
    if(level >= tsr->nmodes) {
        **splits_end = malloc(sizeof ***splits_end);
        spt_CheckOSError(**splits_end == NULL, "SpTns PartSplt");
        (**splits_end)->next = NULL;
        /* Do not free tsr, move it to the destination instead */
        (**splits_end)->tensor = *tsr;
        if(emit_map) {
            /* Keep a snapshot of current inds_low & inds_high */
            (**splits_end)->inds_low = malloc(2 * tsr->nmodes * sizeof (size_t));
            spt_CheckOSError((**splits_end)->inds_low == NULL, "SpTns PartSplt");
            (**splits_end)->inds_high = (**splits_end)->inds_low + tsr->nmodes;
            memcpy((**splits_end)->inds_low, inds_low, tsr->nmodes * sizeof (size_t));
            memcpy((**splits_end)->inds_high, inds_high, tsr->nmodes * sizeof (size_t));
        } else {
            (**splits_end)->inds_low = NULL;
            (**splits_end)->inds_low = NULL;
        }
        *splits_end = &(**splits_end)->next;
        ++*nsplits;
        return 0;
    }

    /* No cuts required at this mode */
    if(nnz_limit_by_mode[level] == 0) {
        if(emit_map) {
            inds_low[level] = tsr->inds[0].data[0];
            inds_high[level] = tsr->inds[0].data[tsr->nnz - 1] + 1;
        }

        spt_RotateMode(tsr);
        sptSparseTensorSortIndex(tsr);
        return spt_SparseTensorPartialSplit(splits_end, nsplits, tsr, nnz_limit_by_mode, index_limit_by_mode, emit_map, inds_low, inds_high, level+1);
    }

    size_t cut_idx = 0;
    size_t cut_low = 0;
    while(cut_low < tsr->nnz) {
        size_t cut_high_est = cut_low + nnz_limit_by_mode[level];
        size_t cut_high;
        if(cut_high_est < tsr->nnz) {
            /* Find a previous step on the index */
            cut_high = spt_FindSplitStep(tsr, cut_high_est, 0);
            if(cut_high <= cut_low) {
                /* Find a next step instead */
                cut_high = spt_FindSplitStep(tsr, cut_high_est, 1);
                fprintf(stderr, "[SpTns PartSplt] cut #%zu size may exceed limit (%zu > %zu)\n", *nsplits+1, cut_high-cut_low, nnz_limit_by_mode[level]);
            }
        } else {
            cut_high = tsr->nnz;
        }
        assert(cut_high > cut_low);

        /* Extract this cut into a new subtensor */
        sptSparseTensor subtsr;
        result = sptNewSparseTensor(&subtsr, tsr->nmodes, tsr->ndims);
        spt_CheckError(result, "SpTns PartSplt", NULL);
        size_t m;
        for(m = 0; m < subtsr.nmodes; ++m) {
            result = sptResizeSizeVector(&subtsr.inds[m], cut_high - cut_low);
            spt_CheckError(result, "SpTns PartSplt", NULL);
            memcpy(subtsr.inds[m].data, &tsr->inds[m].data[cut_low], (cut_high-cut_low) * sizeof (size_t));
        }
        result = sptResizeVector(&subtsr.values, cut_high - cut_low);
        spt_CheckError(result, "SpTns PartSplt", NULL);
        memcpy(subtsr.values.data, &tsr->values.data[cut_low], (cut_high-cut_low) * sizeof (sptScalar));
        subtsr.nnz = cut_high - cut_low;

        if(emit_map) {
            inds_low[level] = subtsr.inds[0].data[0];
            inds_high[level] = subtsr.inds[0].data[subtsr.nnz - 1] + 1;
        }

        spt_RotateMode(&subtsr);
        sptSparseTensorSortIndex(&subtsr);
        result = spt_SparseTensorPartialSplit(splits_end, nsplits, &subtsr, nnz_limit_by_mode, index_limit_by_mode, emit_map, inds_low, inds_high, level+1);
        spt_CheckError(result, "SpTns PartSplt", NULL);

        ++cut_idx;
        cut_low = cut_high;
    }

    /* Freed by the callee */
    sptFreeSparseTensor(tsr);
    return 0;
}

static int spt_SparseTensorIndexSplit(spt_SplitResult ***splits_end, size_t *nsplits, const sptSparseTensor *tsr, const size_t index_limit_by_mode[], int emit_map, size_t inds_low[], size_t inds_high[], size_t level) {
    int result;

    if(level >= tsr->nmodes) {
        **splits_end = malloc(sizeof ***splits_end);
        spt_CheckOSError(**splits_end == NULL, "SpTns IdxSplt");
        (**splits_end)->next = NULL;
        /* Do not free tsr, move it to the destination instead */
        spt_SliceSparseTensor(&(**splits_end)->tensor, tsr, inds_low, inds_high);
        if(emit_map) {
            /* Keep a snapshot of current inds_low & inds_high */
            (**splits_end)->inds_low = malloc(2 * tsr->nmodes * sizeof (size_t));
            spt_CheckOSError((**splits_end)->inds_low == NULL, "SpTns IdxSplt");
            (**splits_end)->inds_high = (**splits_end)->inds_low + tsr->nmodes;
            memcpy((**splits_end)->inds_low, inds_low, tsr->nmodes * sizeof (size_t));
            memcpy((**splits_end)->inds_high, inds_high, tsr->nmodes * sizeof (size_t));
        } else {
            (**splits_end)->inds_low = NULL;
            (**splits_end)->inds_low = NULL;
        }
        *splits_end = &(**splits_end)->next;
        ++*nsplits;
        return 0;
    }

    inds_low[level] = 0;

    while(inds_low[level] < tsr->ndims[level]) {
        inds_high[level] = inds_low[level] + index_limit_by_mode[level];
        result = spt_SparseTensorIndexSplit(splits_end, nsplits, tsr, index_limit_by_mode, emit_map, inds_low, inds_high, level + 1);
        spt_CheckError(result, "SpTns IdxSplt", NULL);
        inds_low[level] += index_limit_by_mode[level];
    }

    return 0;
}

/**
 * A convenient function to get all splits by repeatively calling `spt_SplitSparseTensor`
 *
 * @param[out] splits            Place to store all splits
 * @param[out] nsplits           Place to store the number of actual splits
 * @param[in]  tsr               The tensor to split
 * @param[in]  nnz_limit_by_mode  The size of cuts at each mode, length `tsr->nmodes`
 * @param[in]  emit_map          Whether to emit the index limit of each subtensor
 */
int spt_SparseTensorGetAllSplits(spt_SplitResult **splits, size_t *nsplits, const sptSparseTensor *tsr, const size_t nnz_limit_by_mode[], const size_t index_limit_by_mode[], int emit_map) {
    /* FIXME: index_limit_by_mode is not used yet */
    int result;

    spt_SplitResult **splits_end = splits;

    size_t internal_nsplits;
    if(nsplits == NULL) { nsplits = &internal_nsplits; }
    *nsplits = 0;

    size_t *internal_nnz_limit_by_mode;
    if(nnz_limit_by_mode != NULL) {
        internal_nnz_limit_by_mode = malloc(tsr->nmodes * sizeof (size_t));
        spt_CheckOSError(internal_nnz_limit_by_mode == NULL, "SpTns AllSplts");
        memcpy(internal_nnz_limit_by_mode, nnz_limit_by_mode, tsr->nmodes * sizeof (size_t));
        size_t i;
        for(i = 1; i < tsr->nmodes; ++i) {
            internal_nnz_limit_by_mode[tsr->nmodes - i - 1] *= internal_nnz_limit_by_mode[tsr->nmodes - i];
        }
    } else {
        internal_nnz_limit_by_mode = NULL;
    }

    const size_t *internal_index_limit_by_mode;
    if(index_limit_by_mode != NULL) {
        internal_index_limit_by_mode = index_limit_by_mode;
    } else {
        internal_index_limit_by_mode = calloc(tsr->nmodes, sizeof (size_t));
        spt_CheckOSError(internal_index_limit_by_mode == NULL, "SpTns AllSplts");
    }

    size_t *cut_low, *cut_high;
    if(emit_map) {
        cut_low = malloc(2 * tsr->nmodes * sizeof (size_t));
        spt_CheckOSError(cut_low == NULL, "SpTns AllSplts");
        cut_high = cut_low + tsr->nmodes;
    } else {
        cut_low = NULL;
        cut_high = NULL;
    }

    sptSparseTensor tsr_copy;
    if(nnz_limit_by_mode != NULL) {
        result = sptCopySparseTensor(&tsr_copy, tsr);
        spt_CheckError(result, "SpTns AllSplts", NULL);

        result = spt_SparseTensorPartialSplit(&splits_end, nsplits, &tsr_copy, internal_nnz_limit_by_mode, internal_index_limit_by_mode, emit_map, cut_low, cut_high, 0);
        spt_CheckError(result, "SpTns AllSplts", NULL);
    } else {
        tsr_copy = *tsr;

        result = spt_SparseTensorIndexSplit(&splits_end, nsplits, &tsr_copy, internal_index_limit_by_mode, emit_map, cut_low, cut_high, 0);
        spt_CheckError(result, "SpTns AllSplts", NULL);
    }

    if(emit_map) {
        free(cut_low);
    }
    if(index_limit_by_mode == NULL) {
        free((size_t *) internal_index_limit_by_mode);
    }
    if(nnz_limit_by_mode != NULL) {
        free(internal_nnz_limit_by_mode);
    }

    return 0;
}

/**
 * Free all sub-tensors created by `spt_SparseTensorGetAllSplits`
 */
void spt_SparseTensorFreeAllSplits(spt_SplitResult *splits) {
    while(splits) {
        spt_SplitResult *temp = splits;
        sptFreeSparseTensor(&splits->tensor);
        free(splits->inds_low);
        splits = splits->next;
        free(temp);
    }
}

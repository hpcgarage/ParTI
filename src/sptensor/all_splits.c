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
#include "sort.h"
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

/**
 * In-place split sub-tensors only according to index limitations.
 * TODO: decrease the complexity, only trasverse tsr once.
 * This function is similar to sorting.
 * @param[out] tsr
 * @param[out] nsplits_out
 * @param[out] nnzs_per_split
 */
static int spt_SparseTensorIndexSplitInPlace(
    sptSparseTensor *tsr,
    size_t * nsplits_out,
    sptSizeVector * nnzs_per_split,
    sptSizeVector * inds_low,
    sptSizeVector * inds_high,
    const size_t index_limit_by_mode[]) {

    sptDumpSparseTensor(tsr, 0, stdout);

    int result;
    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t * const ndims = tsr->ndims;
    sptSizeVector * inds = tsr->inds;

    /* Do nothing with an empty tensor */
    if(nnz == 0) {
        sptFreeSparseTensor(tsr);
        return 0;
    }
    sptSizeVector tmp_inds_low;
    result = sptNewSizeVector(&tmp_inds_low, nmodes, nmodes);
    spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);
    result = sptConstantSizeVector(&tmp_inds_low, 0);
    spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);
    sptSizeVector tmp_inds_high;
    result = sptNewSizeVector(&tmp_inds_high, nmodes, nmodes);
    spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);
    result = sptConstantSizeVector(&tmp_inds_high, 0);
    spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);


    size_t nsplits = 0;
    size_t split_nnz = 0;
    int finished = 0;
    size_t mode_level = nmodes -1;

    while (!finished) { // Loop splits
        size_t stsr_loc = 0;
        if(nsplits != 0)
            stsr_loc = nnzs_per_split->data[nsplits-1];

        size_t index;
        for(size_t x=0; x<nnz; ++x) {
            int belong_stsr = 1;
            for(size_t m=0; m<nmodes; ++m) {
                index = inds[m].data[x];
                if(index < tmp_inds_low.data[m] || index >= tmp_inds_low.data[m] + index_limit_by_mode[m]) {
                    belong_stsr = 0;
                    break;
                }
            }
            if(belong_stsr) {
                ++ split_nnz;
                if(x > stsr_loc) {
                    spt_SwapValues(tsr, x, stsr_loc);
                }
                ++ stsr_loc;
            }
        }
        nnzs_per_split->data[nsplits] = split_nnz;
        result = sptAppendSizeVectorWithVector(inds_low, &tmp_inds_low);
        spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);
        printf("\n\nnsplits: %zu\n", nsplits);
        printf("tmp_inds_low:\n");
        sptDumpSizeVector(&tmp_inds_low, stdout);
        for(size_t i=0; i<nmodes; ++i) {
            if(tmp_inds_low.data[i] + index_limit_by_mode[i] < ndims[i])
                tmp_inds_high.data[i] = tmp_inds_low.data[i] + index_limit_by_mode[i] - 1;
            else
                tmp_inds_high.data[i] = ndims[i] - 1;
        }
        result = sptAppendSizeVectorWithVector(inds_high, &tmp_inds_high);
        spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);
        printf("tmp_inds_high:\n");
        sptDumpSizeVector(&tmp_inds_high, stdout);

        // printf("inds_low:\n");
        // sptDumpSizeVector(inds_low, stdout);
        // printf("inds_high:\n");
        // sptDumpSizeVector(inds_high, stdout);
        printf("nnzs_per_split:\n");
        sptDumpSizeVector(nnzs_per_split, stdout);

        /* For next split */
        ++ nsplits;
        split_nnz = 0;
        result = sptAppendSizeVector(nnzs_per_split, split_nnz);
        spt_CheckError(result, "SpTns IdxSpltInPlace", NULL);

        /* Update tmp_inds_low in the reverse order of mode-nmodes-1, ..., 1, 0. */
        while(tmp_inds_low.data[0] + index_limit_by_mode[0] < ndims[0]) {
            if(tmp_inds_low.data[mode_level] + index_limit_by_mode[mode_level] >= ndims[mode_level]) {
                for(size_t i=mode_level; i<nmodes; ++i) {
                    tmp_inds_low.data[i] = 0;
                }
                if(mode_level > 0)
                    -- mode_level;
                else {
                    finished = 1;
                    break;
                }
            } else {
                tmp_inds_low.data[mode_level] += index_limit_by_mode[mode_level];
                finished = 0;
                break;
            }
        }
        printf("mode_level: %lu\n", mode_level);
        printf("tmp_inds_low for next split:\n");
        sptDumpSizeVector(&tmp_inds_low, stdout);
    }   // End while


    sptFreeSizeVector(&tmp_inds_low);
    sptFreeSizeVector(&tmp_inds_high);

    *nsplits_out = nsplits;

    return 0;
}


/**
 * Split sub-tensors into balanced subtensors by considering both nnz number and index limits.
 * @param[out] splits
 * @param[out] nsplits
 * @param[in] tsr: rearranged by spt_SparseTensorIndexSplitInPlace
 * @param[in] nsplits_idxsplit: number of splits from spt_SparseTensorIndexSplitInPlace
 * @param[in] nnzs_per_split: number of nonzeros of each split, generated from spt_SparseTensorIndexSplitInPlace
 * @param[in] nnz_limit: the number limitation  of nonzeros.
 * @param[in] theta: nnz balance threshold.
 */
int spt_SparseTensorNnzSplit(
    spt_SplitResult ***splits,
    size_t *nsplits,
    sptSparseTensor *tsr,
    const size_t nsplits_idxsplit,
    sptSizeVector * nnzs_per_split,
    sptSizeVector * inds_low_idxsplit,
    sptSizeVector * inds_high_idxsplit,
    const size_t nnz_limit,
    const sptScalar theta) {

    int result;
    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t nsplits_out = 0;
    size_t nnz_threshold = (size_t) (nnz_limit * theta);
    sptAssert(nnzs_per_split->len == nsplits_idxsplit);
    size_t sum = 0;
    for(size_t i=0; i<nsplits_idxsplit; ++i)
        sum += nnzs_per_split->data[i];
    sptAssert(sum == nnz);
    sptAssert(inds_low_idxsplit->len == nsplits_idxsplit * nmodes);


    spt_SplitResult * prev_split = NULL;
    size_t split_nnz_begin = 0;
    int num_partial_strs = 0;
    size_t partial_stsr_locs[2];
    size_t partial_stsr_nnz_begin[2];
    sptSparseTensor subtsr; // Just a puppet, don't free it.
    subtsr.nmodes = nmodes;
    subtsr.sortkey = tsr->sortkey;
    subtsr.ndims = tsr->ndims;
    for(size_t s=0; s<nsplits_idxsplit; ++s) {
        size_t split_nnz = nnzs_per_split->data[s];
        if(split_nnz > nnz_threshold && split_nnz <= nnz_limit) {
            ++ nsplits_out;
            **splits = malloc(sizeof ***splits);
            spt_CheckOSError(**splits == NULL, "SpTns NnzSplt");
            (**splits)->next = NULL;
            (**splits)->inds_low = malloc(2 * nmodes * sizeof (size_t));
            spt_CheckOSError((**splits)->inds_low == NULL, "SpTns NnzSplt");
            (**splits)->inds_high = (**splits)->inds_low + nmodes;
            for(size_t i=0; i<nmodes; ++i) {
                (**splits)->inds_low[i] = inds_low_idxsplit->data[s*nmodes+i];
                (**splits)->inds_high[i] = inds_high_idxsplit->data[s*nmodes+i];
            }
            subtsr.nnz = split_nnz;
            sptSizeVector * stsr_inds = (sptSizeVector *)malloc(nmodes * sizeof(sptSizeVector));
            for(size_t i=0; i<nmodes; ++i) {
                result = sptNewSizeVector(&(stsr_inds[i]), split_nnz, split_nnz);
                spt_CheckError(result, "SpTns NnzSplt", NULL);
                for(size_t j=0; j<split_nnz; ++j)
                    stsr_inds[i].data[j] = tsr->inds[i].data[j + split_nnz_begin];
            }
            subtsr.inds = stsr_inds;
            sptVector stsr_vals;
            result = sptNewVector(&stsr_vals, split_nnz, split_nnz);
            spt_CheckError(result, "SpTns NnzSplt", NULL);
            for(size_t j=0; j<split_nnz; ++j)
                stsr_vals.data[j] = tsr->values.data[j + split_nnz_begin];
            subtsr.values = stsr_vals;
            (**splits)->tensor = subtsr;
            prev_split->next = **splits;

        } else if(split_nnz > 0 && split_nnz <= nnz_threshold) {
            /* Combine two sub-tensors as one */
            partial_stsr_locs[num_partial_strs] = s;
            partial_stsr_nnz_begin[num_partial_strs] = split_nnz_begin;
            ++ num_partial_strs;

            if(num_partial_strs == 2) {
                ++ nsplits_out;

                **splits = malloc(sizeof ***splits);
                spt_CheckOSError(**splits == NULL, "SpTns NnzSplt");
                (**splits)->next = NULL;
                (**splits)->inds_low = malloc(2 * nmodes * sizeof (size_t));
                spt_CheckOSError((**splits)->inds_low == NULL, "SpTns NnzSplt");
                (**splits)->inds_high = (**splits)->inds_low + nmodes;
                for(size_t i=0; i<nmodes; ++i) {
                    (**splits)->inds_low[i] =
                    (inds_low_idxsplit->data[partial_stsr_locs[0]*nmodes+i] < inds_low_idxsplit->data[partial_stsr_locs[1]*nmodes+i]) ?
                    inds_low_idxsplit->data[partial_stsr_locs[0]*nmodes+i] : inds_low_idxsplit->data[partial_stsr_locs[1]*nmodes+i];
                    (**splits)->inds_high[i] =
                    (inds_high_idxsplit->data[partial_stsr_locs[0]*nmodes+i] > inds_high_idxsplit->data[partial_stsr_locs[1]*nmodes+i]) ?
                    inds_high_idxsplit->data[partial_stsr_locs[0]*nmodes+i] : inds_high_idxsplit->data[partial_stsr_locs[0]*nmodes+i];
                }

                size_t nnz_1 = nnzs_per_split->data[partial_stsr_locs[0]];
                size_t nnz_2 = nnzs_per_split->data[partial_stsr_locs[1]];
                subtsr.nnz =  nnz_1 + nnz_2;
                sptSizeVector * stsr_inds = (sptSizeVector *)malloc(nmodes * sizeof(sptSizeVector));
                for(size_t i=0; i<nmodes; ++i) {
                    result = sptNewSizeVector(&(stsr_inds[i]), subtsr.nnz, subtsr.nnz);
                    spt_CheckError(result, "SpTns NnzSplt", NULL);
                    for(size_t j=0; j<nnz_1; ++j)
                        stsr_inds[i].data[j] = tsr->inds[i].data[j + partial_stsr_nnz_begin[0]];
                    for(size_t j=0; j<nnz_2; ++j)
                        stsr_inds[i].data[j + nnz_1] = tsr->inds[i].data[j + partial_stsr_nnz_begin[1]];
                }
                subtsr.inds = stsr_inds;
                sptVector stsr_vals;
                result = sptNewVector(&stsr_vals, subtsr.nnz, subtsr.nnz);
                spt_CheckError(result, "SpTns NnzSplt", NULL);
                for(size_t j=0; j<nnz_1; ++j)
                    stsr_vals.data[j] = tsr->values.data[j + partial_stsr_nnz_begin[0]];
                for(size_t j=0; j<nnz_2; ++j)
                    stsr_vals.data[j + nnz_1] = tsr->values.data[j + partial_stsr_nnz_begin[1]];
                subtsr.values = stsr_vals;
                (**splits)->tensor = subtsr;
                prev_split->next = **splits;

                num_partial_strs = 0;
            }
        }
        prev_split = **splits;

        // For the next split
        split_nnz_begin += split_nnz;
    }   // End for loop for splits

    * nsplits = nsplits_out;

    return 0;
}


/**
 * Split sub-tensors into balanced subtensors by considering both nnz number and index limits.
 * @param[out] splits
 * @param[out] nsplits
 */
int spt_SparseTensorBalancedSplit(
    spt_SplitResult **splits,
    size_t *nsplits,
    sptSparseTensor *tsr,
    const size_t nnz_limit,
    const size_t index_limit_by_mode[]) {

    int result;
    size_t const nmodes = tsr->nmodes;

    /* Do nothing with an empty tensor */
    if(tsr->nnz == 0) {
        sptFreeSparseTensor(tsr);
        return 0;
    }
    sptAssert(nnz_limit != 0 && index_limit_by_mode);

    sptSparseTensor tsr_copy;
    result = sptCopySparseTensor(&tsr_copy, tsr);
    spt_CheckError(result, "SpTns AllSplts", NULL);
    sptSizeVector nnzs_per_split;
    result = sptNewSizeVector(&nnzs_per_split, 1, 1);
    spt_CheckError(result, "SpTns AllSplts", NULL);
    sptSizeVector inds_low;
    result = sptNewSizeVector(&inds_low, 0, 0);
    spt_CheckError(result, "SpTns AllSplts", NULL);
    sptSizeVector inds_high;
    result = sptNewSizeVector(&inds_high, 0, 0);
    spt_CheckError(result, "SpTns AllSplts", NULL);
    size_t nsplits_idxsplit = 0;


    /* Reorder the tensor into protential split order. */
    spt_SparseTensorIndexSplitInPlace(&tsr_copy, &nsplits_idxsplit, &nnzs_per_split, &inds_low, &inds_high, index_limit_by_mode);

    // const sptScalar theta = 0.4;
    // /* Consider nnz_limit to actually cut the tensor */
    // spt_SparseTensorNnzSplit(&splits, nsplits, &tsr_copy, nsplits_idxsplit, &nnzs_per_split, &inds_low, &inds_high, nnz_limit, theta);


    sptFreeSizeVector(&nnzs_per_split);
    sptFreeSizeVector(&inds_low);
    sptFreeSizeVector(&inds_high);

    return 0;
}




static int spt_SparseTensorPartialSplit(
    spt_SplitResult ***splits_end,
    size_t *nsplits,
    sptSparseTensor *tsr,
    const size_t nnz_limit_by_mode[],
    const size_t index_limit_by_mode[],
    int emit_map,
    size_t inds_low[],
    size_t inds_high[],
    size_t level) {

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
        return spt_SparseTensorPartialSplit(splits_end, nsplits, tsr, nnz_limit_by_mode, index_limit_by_mode, emit_map,
            inds_low, inds_high, level+1);
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
        result = spt_SparseTensorPartialSplit(splits_end, nsplits, &subtsr, nnz_limit_by_mode, index_limit_by_mode, emit_map,
            inds_low, inds_high, level+1);
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
        spt_SliceSparseTensor(&(**splits_end)->tensor, tsr, inds_low, inds_high);
        if((**splits_end)->tensor.nnz == 0) {
            sptFreeSparseTensor(&(**splits_end)->tensor);
            free(**splits_end);
            **splits_end = NULL;
            return 0;
        }
        if(emit_map) {
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

    for(inds_low[level] = 0; inds_low[level] < tsr->ndims[level]; inds_low[level] += index_limit_by_mode[level]) {
        inds_high[level] = inds_low[level] + index_limit_by_mode[level];
        result = spt_SparseTensorIndexSplit(splits_end, nsplits, tsr, index_limit_by_mode, emit_map, inds_low, inds_high, level + 1);
        spt_CheckError(result, "SpTns IdxSplt", NULL);
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
int spt_SparseTensorGetAllSplits(
    spt_SplitResult **splits,
    size_t *nsplits,
    const sptSparseTensor *tsr,
    const size_t nnz_limit_by_mode[],
    const size_t index_limit_by_mode[],
    int emit_map) {

    int result;
    spt_SplitResult **splits_end = splits;

    // TODO: split a tensor under a specified number of splits.
    size_t internal_nsplits;
    if(nsplits == NULL) { nsplits = &internal_nsplits; }
    else { printf("nsplits has been specified as %lu, which has been supported yet.\n", *nsplits); }
    *nsplits = 0;

    /* Split by specified nnz limitation, while index_limit_by_mode should be NULL. */
    size_t *internal_nnz_limit_by_mode;
    if(nnz_limit_by_mode != NULL) {
        internal_nnz_limit_by_mode = malloc(tsr->nmodes * sizeof (size_t));
        spt_CheckOSError(internal_nnz_limit_by_mode == NULL, "SpTns AllSplts");
        memcpy(internal_nnz_limit_by_mode, nnz_limit_by_mode, tsr->nmodes * sizeof (size_t));
        size_t i;
        for(i = 0; i + 1 < tsr->nmodes; ++i) {
            internal_nnz_limit_by_mode[i + 1] *= internal_nnz_limit_by_mode[i];
        }
    } else {
        internal_nnz_limit_by_mode = NULL;
    }

    /* Split by specified index limitation, while nnz_limit_by_mode should be NULL. */
    const size_t *internal_index_limit_by_mode;
    if(index_limit_by_mode != NULL) {
        internal_index_limit_by_mode = index_limit_by_mode;
    } else {
        internal_index_limit_by_mode = calloc(tsr->nmodes, sizeof (size_t));    // why: allocate space for it?
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
        sptAssert(index_limit_by_mode == NULL); // FIXME: not implemented yet, either set nnz to NULL, or set index to NULL
        result = sptCopySparseTensor(&tsr_copy, tsr);
        spt_CheckError(result, "SpTns AllSplts", NULL);

        result = spt_SparseTensorPartialSplit(&splits_end, nsplits, &tsr_copy, internal_nnz_limit_by_mode, internal_index_limit_by_mode,
            emit_map, cut_low, cut_high, 0);
        spt_CheckError(result, "SpTns AllSplts", NULL);
    } else {
        tsr_copy = *tsr;

        result = spt_SparseTensorIndexSplit(&splits_end, nsplits, &tsr_copy, internal_index_limit_by_mode,
            emit_map, cut_low, cut_high, 0);
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

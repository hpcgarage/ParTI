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
#include <ParTI.h>
#include <stdio.h>
#include "sptensor.h"

struct spt_TagSplitHandle {
    const sptSparseTensor *tsr;
    sptSizeVector cuts_by_mode;
    sptSizeVector partial_low;
    sptSizeVector partial_high;
    sptSizeVector index_step;
    int no_more;
};

/**
 * Split a sparse tensor into several sub-tensors.
 *
 * Call spt_StartSplitSparseTensor to start a split operation.
 * Call spt_SplitSparseTensor to get the next result from this split operation.
 * Call spt_FinishSplitSparseTensor to finish this split operation.
 *
 * @param[out] handle        The handle of this split operation
 * @param[in]  tsr           The sparse tensor to split from
 * @param[in]  cuts_by_mode  The number of cuts at each mode, length `tsr->nmodes`
 */
int spt_StartSplitSparseTensor(spt_SplitHandle *handle, const sptSparseTensor *tsr, const size_t cuts_by_mode[]) {
    int result = 0;

    if(tsr->nnz == 0) {
        spt_CheckError(SPTERR_NO_MORE, "SpTns Start Split", "no splits");
    }

    if(tsr->sortkey != tsr->nmodes-1) {
        spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Start Split", "sortkey != nmodes-1");
    }

    *handle = malloc(sizeof **handle);
    (*handle)->tsr = tsr;
    result = sptNewSizeVector(&(*handle)->cuts_by_mode, tsr->nmodes, tsr->nmodes);
    spt_CheckError(result, "SpTns Start Split", NULL);
    memcpy((*handle)->cuts_by_mode.data, cuts_by_mode, tsr->nmodes * sizeof (size_t));

    result = sptNewSizeVector(&(*handle)->partial_low, 1, tsr->nmodes+1);
    spt_CheckError(result, "SpTns Start Split", NULL);
    result = sptNewSizeVector(&(*handle)->partial_high, 1, tsr->nmodes+1);
    spt_CheckError(result, "SpTns Start Split", NULL);
    result = sptNewSizeVector(&(*handle)->index_step, 0, tsr->nmodes);
    spt_CheckError(result, "SpTns Start Split", NULL);

    (*handle)->partial_low.data[0] = 0;
    (*handle)->partial_high.data[0] = tsr->nnz;

    (*handle)->no_more = 0;

    return result;
}

/**
 * Split a sparse tensor into several sub-tensors.
 *
 * Call spt_StartSplitSparseTensor to start a split operation.
 * Call spt_SplitSparseTensor to get the next result from this split operation.
 * Call spt_FinishSplitSparseTensor to finish this split operation.
 *
 * @param[out] dest    A pointer to an uninitialized sptSparseTensor, to hold the sub-tensor output
 * @param      handle  The handle to this split operation
 * @return             Zero on success, `SPTERR_NO_MORE` when there is no more splits, or any other values to indicate an error
 */
int spt_SplitSparseTensor(sptSparseTensor *dest, spt_SplitHandle handle) {
    int result = 0;

    size_t mode = handle->partial_low.len-1;

    if(handle->no_more) {
        return SPTERR_NO_MORE;
    }

    while(mode < handle->tsr->nmodes) {
        size_t low = handle->partial_low.data[mode];
        size_t high = handle->partial_high.data[mode];
        assert(low < high);

        //fprintf(stderr, "Stage 1, mode=%zu, low=%zu, high=%zu\n", mode, low, high);

        // Count distinct index values on this mode
        size_t last_index = handle->tsr->inds[mode].data[low];
        size_t index_counts = 1;
        //fprintf(stderr, "Stage 1, inds[%zu][%zu] = %zu\n", mode, low, last_index+1);
        size_t i;
        for(i = low; i < high; ++i) {
            if(handle->tsr->inds[mode].data[i] != last_index) {
                //fprintf(stderr, "Stage 1, inds[%zu][%zu] = %zu != %zu\n", mode, i, handle->tsr->inds[mode].data[i]+1, last_index+1);
                ++index_counts;
                last_index = handle->tsr->inds[mode].data[i];
            }
        }

        //fprintf(stderr, "Stage 1, index_counts[%zu] = %zu\n", mode, index_counts);

        // Calculate index step for this mode
        size_t index_step = 1;
        if(index_counts != 0) {
            index_step = (index_counts-1) / handle->cuts_by_mode.data[mode] + 1;
        }
        result = sptAppendSizeVector(&handle->index_step, index_step);
        spt_CheckError(result, "SpTns Split", NULL);

        //fprintf(stderr, "Stage 1, index_step[%zu] = %zu\n", mode, index_step);

        // Set initial cut for this mode
        last_index = handle->tsr->inds[mode].data[low];
        index_counts = 1;
        //fprintf(stderr, "Stage 1, inds[%zu][%zu] = %zu\n", mode, low, last_index+1);
        for(i = low; i < high; ++i) {
            if(handle->tsr->inds[mode].data[i] != last_index) {
                //fprintf(stderr, "Stage 1, inds[%zu][%zu] = %zu != %zu\n", mode, i, handle->tsr->inds[mode].data[i]+1, last_index+1);
                last_index = handle->tsr->inds[mode].data[i];
                if(index_counts == handle->index_step.data[mode]) {
                    break;
                }
                ++index_counts;
            }
        }
        result = sptAppendSizeVector(&handle->partial_low, low);
        spt_CheckError(result, "SpTns Split", NULL);
        result = sptAppendSizeVector(&handle->partial_high, i);
        spt_CheckError(result, "SpTns Split", NULL);

        ++mode;
    }

    // Now we have gone through the initial cutting for all modes
    size_t cut_low = handle->partial_low.data[handle->tsr->nmodes];
    size_t cut_high = handle->partial_high.data[handle->tsr->nmodes];

    //fprintf(stderr,  "Stage 2, cut_low=%zu, cut_high=%zu\n", cut_low, cut_high);

    // Copy values[cut_low] through values[cut_high] into the new tensor
    result = sptNewSparseTensor(dest, handle->tsr->nmodes, handle->tsr->ndims);
    spt_CheckError(result, "SpTns Split", NULL);
    for(mode = 0; mode < handle->tsr->nmodes; ++mode) {
        result = sptResizeSizeVector(&dest->inds[mode], cut_high-cut_low);
        spt_CheckError(result, "SpTns Split", NULL);
        memcpy(dest->inds[mode].data, &handle->tsr->inds[mode].data[cut_low], (cut_high - cut_low) * sizeof (size_t));
    }
    result = sptResizeVector(&dest->values, cut_high-cut_low);
    spt_CheckError(result, "SpTns Split", NULL);
    memcpy(dest->values.data, &handle->tsr->values.data[cut_low], (cut_high - cut_low) * sizeof (sptScalar));
    dest->nnz = cut_high - cut_low;

    // Find the next chunk and return current function
    mode = handle->tsr->nmodes;
    while(mode-- > 0) {
        // Starting from the rest of this mode, to the end of previous mode
        size_t low = handle->partial_high.data[mode+1];
        size_t high = handle->partial_high.data[mode];
        //fprintf(stderr, "Stage 3, mode=%zu, last=%zu, limit=%zu\n", mode, low, high);
        if(low >= high) {
            //fprintf(stderr, "Stage 3, backtracking\n");
            --handle->partial_low.len;
            --handle->partial_high.len;
            --handle->index_step.len;
            continue;
        }

        size_t last_index = handle->tsr->inds[mode].data[low];
        size_t index_counts = 1;
        //fprintf(stderr, "Stage 3, inds[%zu][%zu] = %zu\n", mode, low, last_index+1);
        size_t i;
        for(i = low; i < high; ++i) {
            if(handle->tsr->inds[mode].data[i] != last_index) {
                //fprintf(stderr, "Stage 3, inds[%zu][%zu] = %zu != %zu\n", mode, i, handle->tsr->inds[mode].data[i]+1, last_index+1);
                last_index = handle->tsr->inds[mode].data[i];
                if(index_counts == handle->index_step.data[mode]) {
                    break;
                }
                ++index_counts;
            }
        }
        handle->partial_low.data[mode+1] = low;
        handle->partial_high.data[mode+1] = i;
        //fprintf(stderr, "Stage 3, new_low=%zu, new_high=%zu\n", low, i);
        return 0;
    }

    // Mode should be 0 now, which means unable to find the next chunk
    handle->no_more = 1;
    return 0;
}

/**
 * Split a sparse tensor into several sub-tensors.
 *
 * Call spt_StartSplitSparseTensor to start a split operation.
 * Call spt_SplitSparseTensor to get the next result from this split operation.
 * Call spt_FinishSplitSparseTensor to finish this split operation.
 *
 * @param handle  The handle to this split operation
 */
void spt_FinishSplitSparseTensor(spt_SplitHandle handle) {
    handle->tsr = NULL;
    sptFreeSizeVector(&handle->cuts_by_mode);
    sptFreeSizeVector(&handle->partial_low);
    sptFreeSizeVector(&handle->partial_high);
    sptFreeSizeVector(&handle->index_step);
    free(handle);
}

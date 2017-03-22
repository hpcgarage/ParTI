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

/* TODO:
    Also use block-sorted.
*/

#include <assert.h>
#include <ParTI.h>
#include "sptensor.h"


int spt_ComputeFineSplitParameters(
    size_t * split_nnz_len, // Scalar
    sptSparseTensor * const tsr,
    size_t const R,
    size_t const memwords) 
{
    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t * const ndims = tsr->ndims;

    size_t factor_words = 0;
    for(size_t i=0; i<nmodes; ++i) {
        factor_words += ndims[i];
    }
    factor_words *= R;
    printf("factor_words: %zu\n", factor_words);
    if(memwords <= factor_words) {
        printf("Error: set a larger memory size.\n");
        return -1;
    }

    size_t tensor_words = 0;
    size_t tmp_split_nnz_len = (memwords - factor_words) / (nmodes + 1);
    if(tmp_split_nnz_len > nnz) {
        *split_nnz_len = nnz;
    } else {
        *split_nnz_len = tmp_split_nnz_len;
    }

    return 0;
}



/**
 * A fine-grain split to get all splits by repeatively calling `spt_FineSplitSparseTensorStep`
 *
 * @param[out] splits            Place to store all splits
 * @param[out] nsplits            Place to store the number of total splits
 * @param[in] split_nnz_len            Given the nonzero length of the split (the last split may has smaller number), scalar for fine-grain.
 * @param[in]  tsr               The tensor to split
 */
int spt_FineSplitSparseTensorBatch(
    spt_SplitResult * splits,
    size_t * nnz_split_next,
    const size_t nsplits,
    const size_t split_nnz_len,
    sptSparseTensor * tsr,
    size_t const nnz_split_begin) 
{
    size_t const nnz = tsr->nnz;
    sptAssert(split_nnz_len <= nnz);  

    size_t nnz_ptr_next = 0, nnz_ptr_begin = nnz_split_begin;
    sptAssert( spt_FineSplitSparseTensorStep(&(splits[0]), &nnz_ptr_next, split_nnz_len, tsr, nnz_ptr_begin) == 0 );
    for(size_t s=1; s<nsplits; ++s) {
        nnz_ptr_begin = nnz_ptr_next;
        if(nnz_ptr_begin < nnz) {
            sptAssert( spt_FineSplitSparseTensorStep(&(splits[s]), &nnz_ptr_next, split_nnz_len, tsr, nnz_ptr_begin) == 0 );
            splits[s-1].next = &(splits[s]);
        } else {
            break;
        }
    }
    *nnz_split_next = nnz_ptr_next;

    return 0;
}


/**
 * A fine-grain split to get a sub-tensor
 *
 * @param[out] split            Place to store a split
 * @param[in] split_nnz_len            Given the nonzero length of the split (the last split may has smaller number), scalar for fine-grain.
 * @param[in]  tsr               The tensor to split
 * @param[in] nnz_ptr_begin     The nonzero point to begin the split
 */
int spt_FineSplitSparseTensorStep(
    spt_SplitResult * split,
    size_t * nnz_ptr_next,
    const size_t split_nnz_len,
    sptSparseTensor * tsr,
    const size_t nnz_ptr_begin) 
{
    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    sptSizeVector * const inds = tsr->inds;
    sptVector const values = tsr->values;
    sptAssert(nnz_ptr_begin < nnz);
    size_t i;

    sptSparseTensor substr;
    size_t * subndims = (size_t *)malloc(nmodes * sizeof(size_t));
    for(i=0; i<nmodes; ++i) {
        subndims[i] = tsr->ndims[i];
    }
    /* substr.ndims range is larger than its actual range which indicates by inds_low and inds_high. */
    sptAssert( sptNewSparseTensor(&substr, nmodes, subndims) == 0 );
    free(subndims); // substr.ndims is hard copy.

    size_t * inds_low = (size_t *)malloc(2 * nmodes * sizeof(size_t));
    memset(inds_low, 0, 2 * nmodes * sizeof(size_t));
    size_t * inds_high = inds_low + nmodes;

    size_t subnnz = 0;
    *nnz_ptr_next = (nnz_ptr_begin + split_nnz_len < nnz) ? nnz_ptr_begin + split_nnz_len : nnz;
    size_t tmp_ind;
    for(size_t m=0; m<nmodes; ++m) {
        tmp_ind = inds[m].data[nnz_ptr_begin];
        inds_low[m] = tmp_ind;
        inds_high[m] = tmp_ind;
    }
    ++ subnnz;
    for(i=nnz_ptr_begin+1; i<*nnz_ptr_next; ++i) {
        ++ subnnz;
        for(size_t m=0; m<nmodes; ++m) {
            tmp_ind = inds[m].data[i];
            if(tmp_ind < inds_low[m]) {
                inds_low[m] = tmp_ind;
            }
            if(tmp_ind > inds_high[m]) {
                inds_high[m] = tmp_ind;
            }
        }
    }
    for(size_t m=0; m<nmodes; ++m) {
        ++ inds_high[m];
    }
    
    substr.nnz = subnnz;
    for(size_t m=0; m<nmodes; ++m) {
        substr.inds[m].len = substr.nnz;
        substr.inds[m].cap = substr.nnz;
        substr.inds[m].data = inds[m].data + nnz_ptr_begin; // pointer copy
    }
    substr.values.len = substr.nnz;
    substr.values.cap = substr.nnz;
    substr.values.data = values.data + nnz_ptr_begin; // pointer copy

    split->next = NULL;
    split->tensor = substr;    // pointer copy
    split->inds_low = inds_low;
    split->inds_high = inds_high;

    return 0;
}
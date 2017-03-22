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
#include "math.h"
#include <ParTI.h>
#include "sptensor.h"


/* Assume Ib = Jb = Kb */
int spt_ComputeMediumSplitParameters(
    size_t * split_idx_len, // size: nmodes
    sptSparseTensor * const tsr,
    size_t const R,
    size_t const memwords) 
{
    size_t const nmodes = tsr->nmodes;
    size_t * const ndims = tsr->ndims;

    size_t tmp_split_idx_len = (size_t)((double)memwords / (nmodes * R));
    tmp_split_idx_len = pow(2, (int)(log(tmp_split_idx_len) / log(2)));
    // printf("tmp_split_idx_len: %zu\n", tmp_split_idx_len);

    for(size_t i=0; i<nmodes; ++i) {
        if(tmp_split_idx_len <= ndims[i])
            split_idx_len[i] = tmp_split_idx_len;
        else
            split_idx_len[i] = ndims[i];
    }

    return 0;
}

/**
 * A medium-grain split to get all splits by repeatively calling `spt_MediumSplitSparseTensorStep`
 *
 * @param[out] splits            Place to store all splits
 * @param[out] nsplits            Place to store the number of total splits
 * @param[in] split_idx_len            Given the index lengths for all modes of the split (the last split may has smaller number), an array for medium-grain. Each length should be multiple "b"s.
 * @param[in]  tsr               The tensor to split
 * @param[in] blk_size          The block size for block sorting of tsr
 */
int spt_MediumSplitSparseTensorBatch(
    spt_SplitResult * splits,
    size_t * nnz_split_next,
    size_t const nsplits,
    size_t * const split_idx_len,
    sptSparseTensor * tsr,
    size_t const nnz_split_begin,
    size_t * est_inds_low,
    size_t * est_inds_high) 
{
    size_t const nmodes = tsr->nmodes;
    size_t const * ndims = tsr->ndims;
    for(size_t i=0; i<nmodes; ++i)
        sptAssert(split_idx_len[i] > 0 && split_idx_len[i] <= ndims[i]);

    size_t nnz_ptr_next = 0, nnz_ptr_begin = nnz_split_begin;
    size_t subnnz = 0;
    while (subnnz == 0) {
        for(size_t j=0; j<nmodes; ++j) {
            if(est_inds_low[j] + split_idx_len[j] <= ndims[j])
                est_inds_high[j] = est_inds_low[j] + split_idx_len[j];
            else
                est_inds_high[j] = ndims[j];
        }        

        // printf("est_inds_low:\n");
        // spt_DumpArray(est_inds_low, nmodes, 0, stdout);
        // printf("est_inds_high:\n");
        // spt_DumpArray(est_inds_high, nmodes, 0, stdout);
        
        sptAssert(spt_MediumSplitSparseTensorStep(&(splits[0]), &nnz_ptr_next, &subnnz, split_idx_len, tsr, nnz_ptr_begin, est_inds_low, est_inds_high) == 0 );

        // printf("subnnz: %zu\n", subnnz);

        /* Prepare est_inds_low for the next call */
        for(int i=nmodes-1; i>=0; --i) {
            if(est_inds_high[i] < ndims[i]) {
                est_inds_low[i] += split_idx_len[i];
                for(size_t j=i+1; j<nmodes; ++j) {
                    est_inds_low[j] = 0;
                }
                break;
            }
        }

    } // End while (subnnz == 0)
    // spt_SparseTensorDumpAllSplits(splits, nsplits, stdout);

    int i;
    for(size_t s=1; s<nsplits; ++s) {
        nnz_ptr_begin = nnz_ptr_next;
        // printf("[s %zu] nnz_ptr_begin: %zu\n", s, nnz_ptr_begin);
        subnnz = 0;
        while (subnnz == 0 && nnz_ptr_begin < tsr->nnz) {

            for(size_t j=0; j<nmodes; ++j) {
                if(est_inds_low[j] + split_idx_len[j] <= ndims[j])
                    est_inds_high[j] = est_inds_low[j] + split_idx_len[j];
                else
                    est_inds_high[j] = ndims[j];
            }

            // printf("est_inds_low:\n");
            // spt_DumpArray(est_inds_low, nmodes, 0, stdout);
            // printf("est_inds_high:\n");
            // spt_DumpArray(est_inds_high, nmodes, 0, stdout);

            sptAssert(spt_MediumSplitSparseTensorStep(&(splits[s]), &nnz_ptr_next, &subnnz, split_idx_len, tsr, nnz_ptr_begin, est_inds_low, est_inds_high) == 0 );

            // printf("subnnz: %zu\n", subnnz);

            /* Prepare est_inds_low and est_inds_high for the next call */
            for(i=nmodes-1; i>=0; --i) {
                if(est_inds_high[i] < ndims[i]) {
                    est_inds_low[i] += split_idx_len[i];
                    for(size_t j=i+1; j<nmodes; ++j) {
                        est_inds_low[j] = 0;
                    }
                    break;
                }
            }

            if(i == -1) {
                printf("From indices range -- No more splits, break while loop.\n");
                break;
            }
        }   // End while (subnnz == 0 && nnz_ptr_begin < tsr->nnz)
        splits[s-1].next = &(splits[s]);
        // spt_SparseTensorDumpAllSplits(splits+s, nsplits, stdout);
        if(i == -1) {
            printf("From indices range -- No more splits.\n");
            break;
        }
        if(nnz_ptr_begin >= tsr->nnz) {
            printf("No more nnz to split.\n");
            break;
        }

    }   // Loop nsplits
    *nnz_split_next = nnz_ptr_next;

    return 0;
}


/**
 * A medium-grain split to get a sub-tensor
 *
 * @param[out] split            Place to store a split
  * @param[out] nnz_ptr_next      Place to store the nonzero point for the next split
 * @param[in] split_idx_lens            Given the index lengths for all modes of the split (the last split may has smaller number), an array for medium-grain. Each length should be multiple "b"s.
 * @param[in]  tsr               The tensor to split
 * @param[in] nnz_ptr_begin     The nonzero point to begin the split
 */
int spt_MediumSplitSparseTensorStep(    // In-place
    spt_SplitResult * split,
    size_t * nnz_ptr_next,
    size_t * subnnz,
    size_t * const split_idx_len,
    sptSparseTensor * tsr,
    const size_t nnz_ptr_begin,
    size_t * const est_inds_low,
    size_t * const est_inds_high) 
{
    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t * const ndims = tsr->ndims;
    sptSizeVector * const inds = tsr->inds;
    sptVector const values = tsr->values;
    size_t m;

    /* Copy of subtsr */
    size_t r_ptr;
    size_t tmp_subnnz = 0;
    size_t x;

    for(x=nnz_ptr_begin; x<nnz; ++x) {
        if( spt_SparseTensorCompareIndicesRange(tsr, x, est_inds_low, est_inds_high) == 1 ) 
        {
            if(tmp_subnnz == 0) {
                r_ptr = nnz_ptr_begin;
            } else {
                ++ r_ptr;
            }
            if(r_ptr != x) {
                // printf("x: %zu, r_ptr: %zu\n", x, r_ptr);
                spt_SwapValues(tsr, x, r_ptr);
            }
            ++ tmp_subnnz;
        }
        if(inds[0].data[x] >= est_inds_high[0]) {
            break;
        }
    }
    *subnnz = tmp_subnnz;
    if(tmp_subnnz == 0) {
        return 0;
    }

    // printf("r_ptr: %zu, loop stop at: %zu\n", r_ptr, x-1);
    *nnz_ptr_next = r_ptr + 1;
    // printf("nnz_ptr_begin: %zu, nnz_ptr_next: %zu, subnnz: %zu\n", nnz_ptr_begin, *nnz_ptr_next, *subnnz); fflush(stdout);
    sptAssert(*nnz_ptr_next - nnz_ptr_begin == *subnnz );

    /* Calculate the accurate index range */
    size_t * inds_low = (size_t *)malloc(2 * nmodes * sizeof(size_t));
    memset(inds_low, 0, 2 * nmodes * sizeof(size_t));
    size_t * inds_high = inds_low + nmodes;

    size_t tmp_ind;
    for(size_t m=0; m<nmodes; ++m) {
        tmp_ind = inds[m].data[nnz_ptr_begin];
        inds_low[m] = tmp_ind;
        inds_high[m] = tmp_ind;
    }
    for(size_t i=nnz_ptr_begin+1; i<*nnz_ptr_next; ++i) {
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

    sptSparseTensor substr;
    size_t * subndims = (size_t *)malloc(nmodes * sizeof(size_t));
    /* substr.ndims range may be larger than its actual range which indicates by inds_low and inds_high, except the ndims[mode]. */
    for(size_t i=0; i<nmodes; ++i) {
        subndims[i] = split_idx_len[i];
    }
    sptAssert( sptNewSparseTensor(&substr, nmodes, subndims) == 0 );
    free(subndims); // substr.ndims is hard copy.

    substr.nnz = *subnnz;
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





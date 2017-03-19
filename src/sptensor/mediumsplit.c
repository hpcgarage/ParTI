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

/**
 * A medium-grain split to get all splits by repeatively calling `spt_MediumSplitSparseTensorStep`
 *
 * @param[out] splits            Place to store all splits
 * @param[out] nsplits            Place to store the number of total splits
 * @param[in] split_idx_lens            Given the index lengths for all modes of the split (the last split may has smaller number), an array for medium-grain. Each length should be multiple "b"s.
 * @param[in]  tsr               The tensor to split
 * @param[in] blk_size          The block size for block sorting of tsr
 */
int spt_MediumSplitSparseTensorAll(
    spt_SplitResult ** splits,
    size_t * nsplits,
    size_t * const split_idx_lens,
    sptSparseTensor * tsr,
    int const blk_size) 
{
    size_t const nmodes = tsr->nmodes;
    size_t const * ndims = tsr->ndims;
    for(size_t i=0; i<nmodes; ++i)
        sptAssert(split_idx_lens[i] > 0 && split_idx_lens[i] <= ndims[i]);

    size_t * mode_nsplits = (size_t*)malloc(nmodes * sizeof(size_t));
    memset(mode_nsplits, 0, nmodes * sizeof(size_t));
    size_t max_nsplits = 1;
    for(size_t i=0; i<nmodes; ++i) {
        mode_nsplits[i] = ndims[i] % split_idx_lens[i] == 0 ? ndims[i] / split_idx_lens[i] : ndims[i] / split_idx_lens[i] + 1;
        max_nsplits *= mode_nsplits[i];
    }
    spt_DumpArray(mode_nsplits, nmodes, 0, stdout);
    printf("max_nsplits: %lu\n", max_nsplits);

    *splits = (spt_SplitResult*) malloc((max_nsplits) * sizeof(spt_SplitResult));

    // sptBlockSparseTensor bstr;
    // sptAssert( sptSparseTensorBlockSortIndex(&bstr, tsr, max_nsplits, blk_size) == 0 );  // tsr block-sorted for all modes
    
    size_t rest_loc_size = (size_t)pow(2, nmodes - 2) * blk_size;
    size_t * rest_loc_begin = (size_t *)malloc(rest_loc_size * sizeof(size_t));
    size_t * rest_loc_end = (size_t *)malloc(rest_loc_size * sizeof(size_t));
    memset(rest_loc_begin, 0, rest_loc_size * sizeof(size_t));
    memset(rest_loc_end, 0, rest_loc_size * sizeof(size_t));
    rest_loc_begin[0] = 0;
    rest_loc_end[0] = tsr->nnz;

    size_t * inds_low = (size_t *)malloc(2 * nmodes * sizeof(size_t));
    memset(inds_low, 0, 2 * nmodes * sizeof(size_t));

    sptAssert( spt_MediumSplitSparseTensorStep(&((*splits)[0]), split_idx_lens, tsr, rest_loc_begin, rest_loc_end, rest_loc_size, inds_low) == 0 );
    *nsplits = 1;
    for(size_t s=1; s < max_nsplits; ++s) {
        ++ *nsplits;
        sptAssert( spt_MediumSplitSparseTensorStep(&((*splits)[s]), split_idx_lens, tsr, rest_loc_begin, rest_loc_end, rest_loc_size, inds_low) == 0 );
        (*splits)[s-1].next = &((*splits)[s]);
    }

    free(mode_nsplits);

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
int spt_MediumSplitSparseTensorStep(
    spt_SplitResult * split,
    size_t * const split_idx_lens,
    sptSparseTensor * tsr,
    size_t * rest_loc_begin,
    size_t * rest_loc_end,
    size_t const rest_loc_size,
    size_t * inds_low) 
{
    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t * const ndims = tsr->ndims;
    sptSizeVector * const inds = tsr->inds;
    sptVector const values = tsr->values;
    size_t m;

    printf("\nsplit_idx_lens:\n");
    spt_DumpArray(split_idx_lens, nmodes, 0, stdout);

    sptSparseTensor substr;
    size_t * subndims = (size_t *)malloc(nmodes * sizeof(size_t));
    for(m=0; m<nmodes; ++m) {
        subndims[m] = split_idx_lens[m];
    }
    /* substr.ndims range may be larger than its actual range which indicates by inds_low and inds_high. */
    sptAssert( sptNewSparseTensor(&substr, nmodes, subndims) == 0 );
    free(subndims); // substr.ndims is hard copy.

    size_t * inds_high = inds_low + nmodes;
    for(m=0; m<nmodes; ++m) {
        inds_high[m] = (inds_low[m] + split_idx_lens[m] <= ndims[m]) ? inds_low[m] + split_idx_lens[m] : ndims[m];
    }
    printf("inds_high:\n");
    spt_DumpArray(inds_high, nmodes, 0, stdout);


    size_t subnnz = 0;
    size_t rest_loc_num = 0;
    size_t pre_nnz = 0;
    size_t * tmp_inds = (size_t *)malloc(nmodes * sizeof(size_t));
    // all segments of rest nnzs
    for(size_t r=0; r<rest_loc_size; ++r) {
        size_t nnz_begin = rest_loc_begin[r];
        size_t nnz_end = rest_loc_end[r];
        rest_loc_begin[r] = 0;
        rest_loc_end[r] = 0;
        if(nnz_begin == nnz_end) {
            continue;
        }
        // local nnzs
        pre_nnz = nnz_begin;
        for(size_t x=nnz_begin; x<nnz_end; ++x) {
            for(size_t i=0; i<nmodes; ++i) {
                tmp_inds[i] = inds[m].data[x];
            }
            if( spt_SparseTensorCompareIndicesLT(inds_low, tmp_inds, nmodes) == 1 && spt_SparseTensorCompareIndicesLT(tmp_inds, inds_high, nmodes) == 1 ) {
                ++ subnnz;
            } else {
                if( spt_SparseTensorCompareIndicesLT(tmp_inds, inds_low, nmodes) == 1 ) {
                    printf("Error: cannot happen for a sorted tensor.\n");
                } else {
                    if(rest_loc_begin[r] == 0) {
                        rest_loc_begin[r] = x;
                        rest_loc_num = 0;
                    }
                    ++ rest_loc_num;
                }
            }
        }
    }
    




    /* inds_low for the next call */
    for(int i=nmodes-1; i>=0; --i) {
        if(inds_high[i] < ndims[i]) {
            inds_low[i] += split_idx_lens[i];
            for(size_t j=i+1; j<nmodes; ++j) {
                inds_low[j] = 0;
            }
            break;
        }
    }
    printf("End inds_low:\n");
    spt_DumpArray(inds_low, nmodes, 0, stdout);


    split->next = NULL;
    split->tensor = substr;    // pointer copy
    split->inds_low = (size_t *)malloc(2 * nmodes * sizeof(size_t));
    split->inds_high = inds_low + nmodes;
    for(m=0; m<nmodes; ++m) {
        split->inds_low[m] = inds_low[m];
        split->inds_high[m] = inds_high[m];
    }

    return 0;
}



#if 0
/**
 * A block sorting for tensor
 *
 * @param[in/out]  tsr               The tensor to split
 * @param[in] blk_size    The block size.
 */
int sptSparseTensorBlockSortIndex(
    sptBlockSparseTensor * bstr, 
    sptSparseTensor * tsr, 
    size_t const max_nsplits,
    size_t const blk_size) 
{
    /* Sort once to make tensor in order, to save some work. */
    sptSparseTensorSortIndex(tsr);  // tsr sorted from mode-0, ..., N-1.
    sptDumpSparseTensor(tsr, 0, stdout);

    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t * const ndims = tsr->ndims;
    sptSizeVector * inds = tsr->inds;
    sptVector values = tsr->values;
    size_t const blk_vol = pow(blk_size, nmodes);

    /* Temporary space to store a block tensor */
    size_t ** binds = (size_t **)malloc(nmodes * sizeof(size_t*));
    for(size_t i=0; i<nmodes; ++i) {
        binds[i] = (size_t *)malloc(blk_vol * sizeof(size_t));
    }
    sptScalar * bvals = (sptScalar *)malloc(blk_vol * sizeof(sptScalar));
    size_t * binds_low = (size_t *)malloc(nmodes * sizeof(size_t));
    size_t * binds_high = (size_t *)malloc(nmodes * sizeof(size_t));
    size_t * blkptrs = (size_t *)malloc(max_nsplits * sizeof(size_t));
    memset(blkptrs, 0, max_nsplits * sizeof(size_t));

    size_t rest_loc_size = (size_t)pow(2, nmodes - 2) * blk_size;
    size_t * rest_loc_begin = (size_t *)malloc(rest_loc_size * sizeof(size_t));
    size_t * rest_loc_end = (size_t *)malloc(rest_loc_size * sizeof(size_t));
    memset(rest_loc_begin, 0, rest_loc_size * sizeof(size_t));
    memset(rest_loc_end, 0, rest_loc_size * sizeof(size_t));
    rest_loc_begin[0] = 0;
    rest_loc_end[0] = nnz;

    size_t * tmp_inds = (size_t *)malloc(nmodes * sizeof(size_t));
    size_t blk_nnz = 0;
    size_t nblks = 0;
    size_t rest_loc_num = 0;
    size_t check_mode;


    memset(binds_low, 0, nmodes * sizeof(size_t));
    for(size_t i=0; i<nmodes; ++i)
        binds_high[i] = binds_low[i] + blk_size;


    for(size_t b=0; b<max_nsplits; ++b) {
        check_mode = nmodes - 1;
        while () {
            if(binds_high[check_mode] <= ndims[check_mode]) {
                binds_low[check_mode] = binds_high[check_mode];
                for(size_t j=check_mode+1; j<nmodes; j++) {

                }
                if(binds_high[i] + blk_size > ndims[i])
                    binds_high[i] = ndims[i] - 1;
                else
                    binds_high[i] = binds_high[i] + blk_size - 1;
            } else {
                -- check_mode;
            }
        }



    }

    // all segments of rest nnzs
    for(size_t r=0; r<rest_loc_size; ++r) {
        size_t nnz_begin = rest_loc_begin[r];
        size_t nnz_end = rest_loc_end[r];
        rest_loc_begin[r] = 0;
        rest_loc_end[r] = 0;
        if(nnz_begin == nnz_end) {
            continue;
        }
        // local nnzs
        for(size_t x=nnz_begin; x<nnz_end; ++x) {
            for(size_t i=0; i<nmodes; ++i) {
                tmp_inds[i] = inds[m].data[x];
            }
            if( spt_SparseTensorCompareIndicesLT(binds_low, tmp_inds, nmodes) == 1 && spt_SparseTensorCompareIndicesLT(tmp_inds, binds_high, nmodes) == 1 ) {
                ++ blk_nnz;
            } else {
                if( spt_SparseTensorCompareIndicesLT(tmp_inds, binds_low, nmodes) == 1 ) {
                    printf("Error: cannot happen for a sorted tensor.\n");
                } else {
                    if(rest_loc_begin[r] == 0) {
                        rest_loc_begin[r] = x;
                        rest_loc_num = 0;
                    }
                    ++ rest_loc_num;
                }
            }
        }
    }

    return 0;
}
#endif




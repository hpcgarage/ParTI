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
#include "sptensor.h"


/**
 * A coarse-grain split to get all splits by repeatively calling `spt_CoarseSplitSparseTensorStep`
 *
 * @param[out] splits            Place to store all splits
 * @param[out] nsplits            Place to store the number of total splits
 * @param[in] split_idx_len            Given the index length of the split (the last split may has smaller number), scalar for coarse-grain.
 * @param[in] mode            Specify the mode to split, special for coarse-grain.
 * @param[in]  tsr               The tensor to split
 */
int spt_CoarseSplitSparseTensorAll(
    spt_SplitResult ** splits,
    size_t * nsplits,
    const size_t split_idx_len,
    const size_t mode,
    sptSparseTensor * tsr) 
{
    sptAssert(mode < tsr->nmodes);
    sptAssert(split_idx_len > 0);

    size_t const * ndims = tsr->ndims;

    size_t tmp_nsplits = ndims[mode] % split_idx_len == 0 ? ndims[mode] / split_idx_len : ndims[mode] / split_idx_len + 1;
    *nsplits = tmp_nsplits;
    *splits = (spt_SplitResult*) malloc((*nsplits) * sizeof(spt_SplitResult));

    sptSparseTensorSortIndex(tsr);  // tsr sorted from mode-0, ..., N-1.

    size_t nnz_ptr_next = 0, nnz_ptr_begin = 0;
    sptAssert(spt_CoarseSplitSparseTensorStep(&((*splits)[0]), &nnz_ptr_next, split_idx_len, mode, tsr, nnz_ptr_begin));
    for(size_t s=1; s<*nsplits; ++s) {
        nnz_ptr_begin = nnz_ptr_next;
        sptAssert(spt_CoarseSplitSparseTensorStep(&((*splits)[s]), &nnz_ptr_next, split_idx_len, mode, tsr, nnz_ptr_begin));
        (*splits)[s-1].next = &((*splits)[s]);
    }
    
    return 0;
}


/**
 * A coarse-grain split step to get a sub-tensor.
 *
 * @param[out] splits            Place to store a split
 * @param[out] nnz_ptr_next      Place to store the nonzero point for the next split
 * @param[in] split_idx_len            Given the length of the split (the last split may has smaller number), scalar for coarse-grain.
 * @param[in] mode            Specify the mode to split, special for coarse-grain.
 * @param[in]  tsr               The tensor to split
 * @param[in] nnz_ptr_begin     The nonzero point to begin the split
 */
int spt_CoarseSplitSparseTensorStep(
    spt_SplitResult * split,
    size_t * nnz_ptr_next,
    const size_t split_idx_len,
    const size_t mode,
    const sptSparseTensor * tsr,
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
    subndims[mode] = split_idx_len;
    /* substr.ndims range may be larger than its actual range which indicates by inds_low and inds_high. */
    sptAssert( sptNewSparseTensor(&substr, nmodes, subndims) );

    size_t * inds_low = (size_t *)malloc(2 * nmodes * sizeof(size_t));
    memset(inds_low, 0, 2 * nmodes * sizeof(size_t));
    size_t * inds_high = inds_low + nmodes;


    size_t mode_ind, pre_mode_ind;
    size_t ind_num = 0;
    pre_mode_ind = inds[mode].data[nnz_ptr_begin];
    ++ ind_num;
    for(i=nnz_ptr_begin+1; i<nnz; ++i) {
        mode_ind = inds[mode].data[i];
        if(pre_mode_ind != mode_ind) {
            ++ ind_num;
            pre_mode_ind = mode_ind;
        }
        if(ind_num > split_idx_len) {
            break;
        }
    }
    *nnz_ptr_next = i;

    size_t tmp_ind;
    for(size_t m=0; m<nmodes; ++m) {
        tmp_ind = inds[m].data[nnz_ptr_begin];
        inds_low[m] = tmp_ind;
        inds_high[m] = tmp_ind;
    }
    for(i=nnz_ptr_begin+1; i<*nnz_ptr_next; ++i) {
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

    substr.nnz = *nnz_ptr_next - nnz_ptr_begin;
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


#if 1
int sptCoarseSplitSparseTensor(sptSparseTensor *tsr, const int num, sptSparseTensor *cs_splits) 
{
    assert(num > 1);

    size_t const nmodes = tsr->nmodes;
    size_t const nnz = tsr->nnz;
    size_t const * ndims = tsr->ndims;
    sptSizeVector * inds = tsr->inds;
    sptVector values = tsr->values;

    sptSparseTensorSortIndex(tsr);

    size_t * csnnz = (size_t *)malloc(num * sizeof(size_t));
    memset(csnnz, 0, num * sizeof(size_t));
    size_t aver_nnz = nnz / num;


    size_t ** csndims = (size_t**)malloc(num* sizeof(size_t*));
    for(int n=0; n<num; ++n) {
      csndims[n] = (size_t*)malloc(nmodes * sizeof(size_t));
      memset(csndims[n], 0, nmodes * sizeof(size_t));
    }
    size_t * slice_nnzs = (size_t *)malloc(ndims[0] * sizeof(size_t));
    memset(slice_nnzs, 0, ndims[0] * sizeof(size_t));
    for(size_t i=0; i<nnz; ++i) {
        size_t tmp_ind = inds[0].data[i];
        ++ slice_nnzs[tmp_ind];
    }
    // printf("slice_nnzs:\n");
    // for(size_t n=0; n<ndims[0]; ++n) {
    //     printf("%lu ", slice_nnzs[n]);
    // }
    // printf("\n");

    int j = 0;
    for(size_t i=0; i<ndims[0]; ++i) {
        if(csnnz[j] < aver_nnz || j == num - 1) {
            csnnz[j] += slice_nnzs[i];
            ++ csndims[j][0];
        } else {
            ++ j;
            csnnz[j] = slice_nnzs[i];
            csndims[j][0] = 1;
        }
    }
    printf("csnnz:\n");
    for(int n=0; n<num; ++n) {
        printf("%ld ", csnnz[n]);
    }
    printf("\n");
    assert(j == num-1);

    for(int n=0; n<num; ++n) {
      for(size_t m=1; m<nmodes; ++m) {
        csndims[n][m] = ndims[m];
      }
    }

    size_t * nnz_loc = (size_t*)malloc(num*sizeof(size_t));
    memset(nnz_loc, 0, num*sizeof(size_t));
    for(int n=1; n<num; ++n) {
      nnz_loc[n] = nnz_loc[n-1] + csnnz[n-1];
    }

    for(int n=0; n<num; ++n) {
      sptNewSparseTensor(cs_splits+n, nmodes, csndims[n]);
      cs_splits[n].nnz = csnnz[n];
      for(size_t m=0; m<nmodes; ++m) {
        cs_splits[n].inds[m].len = csnnz[n];
        cs_splits[n].inds[m].cap = csnnz[n];
        cs_splits[n].inds[m].data = inds[m].data + nnz_loc[n];
      }
      cs_splits[n].values.len = csnnz[n];
      cs_splits[n].values.cap = csnnz[n];
      cs_splits[n].values.data = values.data + nnz_loc[n];
    }

    // sptDumpSparseTensor(tsr, 0, stdout);
    // printf("\n");
    // for(int n=0; n<num; ++n) {
    //     sptDumpSparseTensor(cs_splits+n, 0, stdout);
    //     printf("\n");
    // }

    free(csnnz);
    free(nnz_loc);
    for(int n=0; n<num; ++n)
        free(csndims[n]);
    free(csndims);
    free(slice_nnzs);

    return 0;
}
#endif
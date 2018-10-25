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

#include <ParTI.h>
#include "sptensor.h"
#include <stdlib.h>
#include <string.h>

/**
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    sptIndex i;
    int result;
    tsr->nmodes = nmodes;
    tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        tsr->sortorder[i] = i;
    }
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewIndexVector(&tsr->inds[i], 0, 0);
        spt_CheckError(result, "SpTns New", NULL);
    }
    result = sptNewValueVector(&tsr->values, 0, 0);
    spt_CheckError(result, "SpTns New", NULL);
    return 0;
}

/**
 * Copy a sparse tensor
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src, int const nt) {
    sptIndex i;
    int result;
    dest->nmodes = src->nmodes;
    dest->sortorder = malloc(src->nmodes * sizeof src->sortorder[0]);
    memcpy(dest->sortorder, src->sortorder, src->nmodes * sizeof src->sortorder[0]);
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SpTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopyIndexVector(&dest->inds[i], &src->inds[i], nt);
        spt_CheckError(result, "SpTns Copy", NULL);
    }
    result = sptCopyValueVector(&dest->values, &src->values, nt);
    spt_CheckError(result, "SpTns Copy", NULL);
    return 0;
}

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    sptIndex i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->sortorder);
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeValueVector(&tsr->values);
    tsr->nmodes = 0;
    tsr->nmodes = 0;
}


double SparseTensorFrobeniusNormSquared(sptSparseTensor const * const spten) 
{
  double norm = 0;
  sptValue const * const restrict vals = spten->values.data;
  
#ifdef PARTI_USE_OPENMP
  #pragma omp parallel for reduction(+:norm)
#endif
  for(sptNnzIndex n=0; n < spten->nnz; ++n) {
    norm += vals[n] * vals[n];
  }
  return norm;
}


int spt_DistSparseTensor(sptSparseTensor * tsr,
    int const nthreads,
    sptNnzIndex * const dist_nnzs,
    sptIndex * dist_nrows) {

    sptNnzIndex global_nnz = tsr->nnz;
    sptNnzIndex aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, nthreads*sizeof(sptNnzIndex));
    memset(dist_nrows, 0, nthreads*sizeof(sptIndex));

    sptSparseTensorSortIndex(tsr, 0);
    sptIndex * ind0 = tsr->inds[0].data;

    int ti = 0;
    dist_nnzs[0] = 1;
    dist_nrows[0] = 1;
    for(sptNnzIndex x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ dist_nnzs[ti];
        } else if (ind0[x] > ind0[x-1]) {
            if(dist_nnzs[ti] < aver_nnz || ti == nthreads-1) {
                ++ dist_nnzs[ti];
                ++ dist_nrows[ti];
            } else {
                ++ ti;
                ++ dist_nnzs[ti];
                ++ dist_nrows[ti];
            }
        } else {
            spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }

    return 0;

}


int spt_DistSparseTensorFixed(sptSparseTensor * tsr,
    int const nthreads,
    sptNnzIndex * const dist_nnzs,
    sptNnzIndex * dist_nrows) {

    sptNnzIndex global_nnz = tsr->nnz;
    sptNnzIndex aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, nthreads*sizeof(sptNnzIndex));

    sptSparseTensorSortIndex(tsr, 0);
    sptIndex * ind0 = tsr->inds[0].data;

    int ti = 0;
    dist_nnzs[0] = 1;
    for(sptNnzIndex x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ dist_nnzs[ti];
        } else if (ind0[x] > ind0[x-1]) {
            if(dist_nnzs[ti] < aver_nnz || ti == nthreads-1) {
                ++ dist_nnzs[ti];
            } else {
                ++ ti;
                ++ dist_nnzs[ti];
            }
        } else {
            spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }

    return 0;
}



/**
 * Shuffle all indices.
 *
 * @param[in] tsr tensor to be shuffled
 * @param[out] map_inds is the renumbering mapping
 *
 */
void sptSparseTensorShuffleIndices(sptSparseTensor *tsr, sptIndex ** map_inds) {
    /* Renumber nonzero elements */
    sptIndex tmp_ind;
    for(sptNnzIndex z = 0; z < tsr->nnz; ++z) {
        for(sptIndex m = 0; m < tsr->nmodes; ++m) {
            tmp_ind = tsr->inds[m].data[z];
            tsr->inds[m].data[z] = map_inds[m][tmp_ind];
        }
    }
    
}

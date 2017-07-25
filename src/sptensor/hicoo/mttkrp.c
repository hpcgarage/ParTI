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
#include "hicoo.h"


/**
 * Matriced sparse tensor in HiCOO format times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the HiCOO sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 */
int sptMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    sptVector * scratch) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptNnzIndex const nnz = hitsr->nnz;
    sptIndex const * const ndims = hitsr->ndims;
    sptScalar const * const vals = hitsr->values.data;
    sptIndex const nmats = nmodes - 1;
    sptIndex const stride = mats[0]->stride;
    int result = 0;

    //  for(size_t m=0; m<nmodes+1; ++m) {
    //      printf("mats %zu\n", m); fflush(stdout);
    //      sptDumpMatrix(mats[m], stdout);
    //  }
    //  printf("mats_order:\n"); fflush(stdout);
    //  sptDumpSizeVector(mats_order, stdout);

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptMatrix * const M = mats[nmodes];
    sptScalar * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(sptScalar));

    sptIndex * kernel_coord = (sptIndex*)malloc(nmodes * sizeof(*kernel_coord));

    sptNnzIndex kptr_begin = 0, kptr_end;
    /* First kernel base coordinates */
    result = sptGetKernelCoordinates(kernel_coord, nmodes, k, hitsr->kstrides);
    spt_CheckError(result, "CPU  HiCOO SpTns MTTKRP", NULL);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len; ++k) {
        /* Get a valid kernel pointer, except the first kernel */
        if(hitsr->kptr[k] > 0) {
            kptr_end = hitsr->kptr[k];
        }
        kptr_begin = kptr_end;
        /* Kernel base coordinates */
        result = sptGetKernelCoordinates(kernel_coord, nmodes, k, hitsr->kstrides);
        spt_CheckError(result, "CPU  HiCOO SpTns MTTKRP", NULL);
    }

    for(size_t x=0; x<nnz; ++x) {

        size_t times_mat_index = mats_order[1];
        sptMatrix * times_mat = mats[times_mat_index];
        size_t * times_inds = X->inds[times_mat_index].data;
        size_t tmp_i = times_inds[x];
        for(size_t r=0; r<R; ++r) {
            scratch->data[r] = times_mat->values[tmp_i * stride + r];
        }

        for(size_t i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            for(size_t r=0; r<R; ++r) {
                scratch->data[r] *= times_mat->values[tmp_i * stride + r];
            }
        }

        sptScalar const entry = vals[x];
        size_t const mode_i = mode_ind[x];
        for(size_t r=0; r<R; ++r) {
            mvals[mode_i * stride + r] += entry * scratch->data[r];
        }
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CPU  HiCOO SpTns MTTKRP");
    sptFreeTimer(timer);

    return 0;
}

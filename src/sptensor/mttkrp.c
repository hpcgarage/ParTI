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

int sptMTTKRP_3D(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);

/**
 * Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 */
int sptMTTKRP(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode) {

    sptIndex const nmodes = X->nmodes;

    if(nmodes == 3) {
        sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
        return 0;
    }

    sptNnzIndex const nnz = X->nnz;
    sptIndex const * const ndims = X->ndims;
    sptValue const * const restrict vals = X->values.data;
    sptIndex const stride = mats[0]->stride;
    sptValueVector scratch;  // Temporary array

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptIndex const * const restrict mode_ind = X->inds[mode].data;
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    sptNewValueVector(&scratch, R, R);
    sptConstantValueVector(&scratch, 0);


    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(sptNnzIndex x=0; x<nnz; ++x) {

        sptIndex times_mat_index = mats_order[1];
        sptMatrix * times_mat = mats[times_mat_index];
        sptIndex * times_inds = X->inds[times_mat_index].data;
        sptIndex tmp_i = times_inds[x];
        sptValue const entry = vals[x];
        for(sptIndex r=0; r<R; ++r) {
            scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
        }

        for(sptIndex i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            for(sptIndex r=0; r<R; ++r) {
                scratch.data[r] *= times_mat->values[tmp_i * stride + r];
            }
        }

        sptIndex const mode_i = mode_ind[x];
        for(sptIndex r=0; r<R; ++r) {
            mvals[mode_i * stride + r] += scratch.data[r];
        }
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    
    sptFreeTimer(timer);
    sptFreeValueVector(&scratch);

    return 0;
}


int sptMTTKRP_3D(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode) 
{
    sptIndex const nmodes = X->nmodes;
    sptNnzIndex const nnz = X->nnz;
    sptIndex const * const ndims = X->ndims;
    sptValue const * const restrict vals = X->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    sptAssert(nmodes ==3);
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }
    

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptIndex const * const restrict mode_ind = X->inds[mode].data;
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(sptValue));

    sptIndex times_mat_index_1 = mats_order[1];
    sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    sptIndex times_mat_index_2 = mats_order[2];
    sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;

    sptIndex mode_i;
    sptIndex tmp_i_1, tmp_i_2;
    sptValue entry;

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);
    for(sptNnzIndex x=0; x<nnz; ++x) {
        mode_i = mode_ind[x];
        tmp_i_1 = times_inds_1[x];
        tmp_i_2 = times_inds_2[x];
        entry = vals[x];

        for(sptIndex r=0; r<R; ++r) {
            mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
        }
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    sptFreeTimer(timer);

    return 0;
}

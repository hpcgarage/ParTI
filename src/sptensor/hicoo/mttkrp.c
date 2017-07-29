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

int sptMTTKRPHiCOO_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptMTTKRPHiCOO_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);

/**
 * Matriced sparse tensor in HiCOO format times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  hitsr    the HiCOO sparse tensor input
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
    sptIndex const mode) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptMTTKRPHiCOO_3D(hitsr, mats, mats_order, mode) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;
    sptValueVector scratch;  // Temporary array

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
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    sptNewValueVector(&scratch, R, R);

    sptIndex * block_coord = (sptIndex*)malloc(nmodes * sizeof(*block_coord));
    sptIndex * ele_coord = (sptIndex*)malloc(nmodes * sizeof(*ele_coord));


    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
            for(sptIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                /* Element indices */
                for(sptIndex m=0; m<nmodes; ++m)
                    ele_coord[m] = (block_coord[m] << hitsr->sb_bits) + hitsr->einds[m].data[z];

                /* Multiply the 1st matrix */
                sptIndex times_mat_index = mats_order[1];
                sptMatrix * times_mat = mats[times_mat_index];
                sptIndex tmp_i = ele_coord[times_mat_index];
                sptValue const entry = vals[z];
                for(sptIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(sptIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    times_mat = mats[times_mat_index];
                    tmp_i = ele_coord[times_mat_index];
                    for(sptIndex r=0; r<R; ++r) {
                        scratch.data[r] *= times_mat->values[tmp_i * stride + r];
                    }
                }

                sptIndex const mode_i = ele_coord[mode];
                for(sptIndex r=0; r<R; ++r) {
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


    free(block_coord);
    free(ele_coord);
    sptFreeValueVector(&scratch);

    return 0;
}


int sptMTTKRPHiCOO_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    sptAssert(nmodes ==3);
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
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    /* block_coord is reused, no need to store ele_coord for 3D tensors */
    sptBlockIndex * block_coord = (sptBlockIndex*)malloc(nmodes * sizeof(*block_coord));

    sptIndex mode_i;
    sptIndex tmp_i_1, tmp_i_2;
    sptValue entry;

    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
            for(sptIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                mode_i = (block_coord[mode] << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                tmp_i_1 = (block_coord[times_mat_index_1] << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = (block_coord[times_mat_index_2] << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                entry = vals[z];
                for(sptIndex r=0; r<R; ++r) {
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


    free(block_coord);

    return 0;
}


int sptMTTKRPHiCOO_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptMTTKRPHiCOO_3D_MatrixTiling(hitsr, mats, mats_order, mode) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;
    sptValueVector scratch;  // Temporary array

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
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    sptNewValueVector(&scratch, R, R);

    sptValue ** blocked_times_mat = (sptValue**)malloc(nmodes * sizeof(*blocked_times_mat));

    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
            for(sptIndex m=0; m<nmodes; ++m)
                blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {

                /* Multiply the 1st matrix */
                sptIndex times_mat_index = mats_order[1];
                sptElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                sptValue const entry = vals[z];
                for(sptIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(sptIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    for(sptIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][tmp_i * stride + r];
                    }
                }

                sptElementIndex const mode_i = hitsr->einds[mode].data[z];
                for(sptIndex r=0; r<R; ++r) {
                    blocked_mvals[mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


    free(blocked_times_mat);
    sptFreeValueVector(&scratch);

    return 0;
}

int sptMTTKRPHiCOO_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    sptAssert(nmodes ==3);
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
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    sptElementIndex mode_i;
    sptElementIndex tmp_i_1, tmp_i_2;
    sptValue entry;
    sptValue * blocked_mvals;
    sptValue * blocked_times_mat_1;
    sptValue * blocked_times_mat_2;

    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                mode_i = hitsr->einds[mode].data[z];
                tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                entry = vals[z];

                for(sptIndex r=0; r<R; ++r) {
                    blocked_mvals[mode_i * stride + r] += entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}

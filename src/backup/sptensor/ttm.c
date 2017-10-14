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

int sptTTMHiCOO_3D_MatrixTiling(
    sptSparseTensorHiCOO * const Y,
    sptSparseTensorHiCOO const * const X,
    sptRankMatrix * U,     // mats[nmodes] as temporary space.
    sptIndex const mode);

/**
 * Matriced sparse tensor in HiCOO format times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] Y    the HiCOO sparse tensor output
 * @param[in]  X    the HiCOO sparse tensor input
 * @param[in]  U    a dense matrix
 * @param[in]  mode   the mode on which the MTTKRP is performed
 *
 * Only do product in the last mode. Rank should < 128, to make sure the blocks of Y is the same with the blocks of X.
 */
int sptTTMHiCOO_MatrixTiling(
    sptSparseTensorHiCOO * const Y,
    sptSparseTensorHiCOO const * const X,
    sptRankMatrix * U,     // mats[nmodes] as temporary space.
    sptIndex const mode)
{
    sptIndex const nmodes = X->nmodes;
    sptAssert(mode == nmodes - 1);

    if(nmodes == 3) {
        sptAssert(sptTTMHiCOO_3D_MatrixTiling(Y, X, U, mode) == 0);
        return 0;
    }

    // TODO: add for nmodes

    return 0;
}

int sptTTMHiCOO_3D_MatrixTiling(
    sptSparseTensorHiCOO * const Y,
    sptSparseTensorHiCOO const * const X,
    sptRankMatrix * U,     // mats[nmodes] as temporary space.
    sptIndex const mode) 
{
    #if 0
    sptIndex const nmodes = X->nmodes;
    sptIndex const * const ndims = X->ndims;
    sptValue const * const restrict vals = X->values.data;
    sptElementIndex const stride = U->stride;

    /* Check the mats. */
    sptAssert(nmodes == 3);
    if(U->nrows != ndims[mode]) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns TTM", "U->nrows != ndims[mode]");
    }

    sptIndex const tmpI = U->nrows;
    sptElementIndex const R = U->ncols;
    sptValue * blocked_mat;
    sptNnzIndex nnz = 0;

    /* Loop kernels */
    for(sptIndex k=0; k<X->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = X->kptr.data[k];
        sptNnzIndex kptr_end = X->kptr.data[k+1];

        /* Append the first block indices */
        sptIndex blk_i_old = X->binds[0].data[kptr_begin];
        sptIndex blk_j_old = X->binds[1].data[kptr_begin];
        sptAppendBlockIndexVector(&Y->binds[0], X->binds[0].data[b]);
        sptAppendBlockIndexVector(&Y->binds[1], X->binds[1].data[b]);
        sptAppendBlockIndexVector(&Y->binds[2], 0);

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            blocked_mat = U->values + (X->binds[mode].data[b] << X->sb_bits) * stride;
            sptIndex blk_i = X->binds[0].data[b];
            sptIndex blk_j = X->binds[1].data[b];
            if(blk_i != blk_i_old || blk_j != blk_j_old) {

            }


            sptNnzIndex bptr_begin = X->bptr.data[b];
            sptNnzIndex bptr_end = X->bptr.data[b+1];

            /* Append the first element indices */
            sptElementIndex ele_i_old = X->einds[0].data[bptr_begin];
            sptElementIndex ele_j_old = X->einds[1].data[bptr_begin];
            sptAppendElementIndexVector(&Y->einds[0], ele_i_old);
            sptAppendElementIndexVector(&Y->einds[1], ele_j_old);
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                ++ nnz;
                sptElementIndex mode_k = X->einds[mode].data[z];
                sptElementIndex ele_i = X->einds[0].data[z];
                sptElementIndex ele_j = X->einds[1].data[z];
                sptValue entry = vals[z];

                if(ele_i != ele_i_old || ele_j != ele_j_old) {
                    sptAppendElementIndexVector(&Y->einds[0], ele_i);
                    sptAppendElementIndexVector(&Y->einds[1], ele_j);
                }
                for(sptElementIndex r=0; r<R; ++r) {
                    sptAppendElementIndexVector(&Y->einds[1], r);
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += entry * 
                        blocked_mat[(sptBlockMatrixIndex)mode_k * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    sptAppendNnzIndexVector(&Y->bptr, nnz);
    #endif
    return 0;
}




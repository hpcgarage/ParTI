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
#include <omp.h>

// TODO: add a no atomic version function

int sptOmpMTTKRPHiCOOKernels(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRPHiCOOKernels_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRPHiCOOBlocks(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb);
int sptOmpMTTKRPHiCOOBlocks_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb);
int sptOmpMTTKRPHiCOOKernelsBlocks(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOOKernelsBlocks_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);


int sptOmpMTTKRPHiCOOKernels_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRPHiCOOKernels_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRPHiCOOBlocks_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb);
int sptOmpMTTKRPHiCOOBlocks_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb);
int sptOmpMTTKRPHiCOOKernelsBlocks_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOOKernesBlocks_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);


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
int sptOmpMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb)
{
    if(tk > 1 && tb == 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernels(hitsr, mats, mats_order, mode, tk) == 0);
    } else if(tk == 1 && tb > 1) {
        sptAssert(sptOmpMTTKRPHiCOOBlocks(hitsr, mats, mats_order, mode, tb) == 0);
    } else if(tk > 1 && tb > 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernelsBlocks(hitsr, mats, mats_order, mode, tk, tb) == 0);
    } else if(tk == 1 && tb == 1) {
        printf("Should specify sequetial MTTKRP.\n");
        return -1;
    }

    return 0;
}


int sptOmpMTTKRPHiCOO_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb)
{
    if(tk > 1 && tb == 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernels_MatrixTiling(hitsr, mats, mats_order, mode, tk) == 0);
    } else if(tk == 1 && tb > 1) {
        sptAssert(sptOmpMTTKRPHiCOOBlocks_MatrixTiling(hitsr, mats, mats_order, mode, tb) == 0);
    } else if(tk > 1 && tb > 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernelsBlocks_MatrixTiling(hitsr, mats, mats_order, mode, tk, tb) == 0);
    } else if(tk == 1 && tb == 1) {
        printf("Should specify sequetial MTTKRP with -d -2.\n");
        return -1;
    }

    return 0;
}


int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb)
{
    if(tk > 1 && tb == 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled(hitsr, mats, mats_order, mode, tk) == 0);
    } else {
        printf("Haven't support block parallelism.\n");
        return -1;
    }

    return 0;
}


int sptOmpMTTKRPHiCOOKernels(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOKernels_3D(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptMatrix * const M = mats[nmodes];
    sptValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        /* Allocate thread-private data */
        sptIndex * block_coord = (sptIndex*)malloc(nmodes * sizeof(*block_coord));
        sptIndex * ele_coord = (sptIndex*)malloc(nmodes * sizeof(*ele_coord));
        sptValueVector scratch; // Temporary array
        sptNewValueVector(&scratch, R, R);

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];        

        /* Loop blocks in a kernel */
        for(sptNnzIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
            for(sptIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptNnzIndex z=bptr_begin; z<bptr_end; ++z) {
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
                // omp_set_lock(&lock);
                for(sptIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
                // omp_unset_lock(&lock);
            }   // End loop entries
        }   // End loop blocks

        /* Free thread-private space */
        free(block_coord);
        free(ele_coord);
        sptFreeValueVector(&scratch);
    }   // End loop kernels

    // omp_destroy_lock(&lock);

    return 0;
}


int sptOmpMTTKRPHiCOOKernels_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
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


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {
            sptBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
            sptBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
            sptBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                sptIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                sptIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                sptIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                sptValue entry = vals[z];
                for(sptIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks
    }   // End loop kernels

    return 0;
}



int sptOmpMTTKRPHiCOOKernels_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOKernels_3D_MatrixTiling(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        /* Allocate thread-private data */
        sptValue ** blocked_times_mat = (sptValue**)malloc(nmodes * sizeof(*blocked_times_mat));
        sptValueVector scratch; // Temporary array
        sptNewValueVector(&scratch, R, R);

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];        

        /* Loop blocks in a kernel */
        for(sptNnzIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Blocked matrices */
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
                for(sptElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(sptIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    for(sptElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                sptElementIndex const mode_i = hitsr->einds[mode].data[z];
                for(sptElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

        /* Free thread-private space */
        free(blocked_times_mat);
        sptFreeValueVector(&scratch);
    }   // End loop kernels

    return 0;
}


int sptOmpMTTKRPHiCOOKernels_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

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
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            sptValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                sptElementIndex mode_i = hitsr->einds[mode].data[z];
                sptElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                sptElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                sptValue entry = vals[z];

                for(sptElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += entry * 
                        blocked_times_mat_1[(sptBlockMatrixIndex)tmp_i_1 * stride + r] * 
                        blocked_times_mat_2[(sptBlockMatrixIndex)tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}



int sptOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);
    sptIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    sptIndexVector * restrict kschr_mode = hitsr->kschr[mode];

    /* Loop parallel iterations */
    for(sptIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        /* Loop kernels */
        #pragma omp parallel for num_threads(tk)
        for(sptIndex k=0; k<num_kernel_dim; ++k) {

            if(i >= kschr_mode[k].len) continue;
            sptIndex kptr_loc = kschr_mode[k].data[i];
            sptNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            sptNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Allocate thread-private data */
            sptValue ** blocked_times_mat = (sptValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            sptValueVector scratch; // Temporary array
            sptNewValueVector(&scratch, R, R);       

            /* Loop blocks in a kernel */
            for(sptNnzIndex b=kptr_begin; b<kptr_end; ++b) {
                /* Blocked matrices */
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
                    for(sptElementIndex r=0; r<R; ++r) {
                        scratch.data[r] = entry * blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                    }
                    /* Multiply the rest matrices */
                    for(sptIndex m=2; m<nmodes; ++m) {
                        times_mat_index = mats_order[m];
                        tmp_i = hitsr->einds[times_mat_index].data[z];
                        for(sptElementIndex r=0; r<R; ++r) {
                            scratch.data[r] *= blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                        }
                    }

                    sptElementIndex const mode_i = hitsr->einds[mode].data[z];
                    for(sptElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                    }
                }   // End loop entries
            }   // End loop blocks

            /* Free thread-private space */
            free(blocked_times_mat);
            sptFreeValueVector(&scratch);
        }   // End loop kernels
    }   // End loop iterations

    return 0;
}



int sptOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

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
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);
    sptIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    sptIndexVector * restrict kschr_mode = hitsr->kschr[mode];
    // printf("nkiters: %u, num_kernel_dim: %u\n", hitsr->nkiters[mode], num_kernel_dim);


    /* Loop parallel iterations */
    for(sptIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        /* Loop kernels */
        #pragma omp parallel for num_threads(tk)
        for(sptIndex k=0; k<num_kernel_dim; ++k) {
            if(i >= kschr_mode[k].len) {
                // printf("i: %u, k: %u\n", i, k);
                continue;
            }
            sptIndex kptr_loc = kschr_mode[k].data[i];
            sptNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            sptNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Loop blocks in a kernel */
            for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

                sptValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
                sptValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
                sptValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

                sptNnzIndex bptr_begin = hitsr->bptr.data[b];
                sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
                /* Loop entries in a block */
                for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                    
                    sptElementIndex mode_i = hitsr->einds[mode].data[z];
                    sptElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                    sptElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                    sptValue entry = vals[z];

                    for(sptElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += entry * 
                            blocked_times_mat_1[(sptBlockMatrixIndex)tmp_i_1 * stride + r] * 
                            blocked_times_mat_2[(sptBlockMatrixIndex)tmp_i_2 * stride + r];
                    }
                    
                }   // End loop entries
            }   // End loop blocks

        }   // End loop kernels
    }   // End loop iterations

    return 0;
}




int sptOmpMTTKRPHiCOOBlocks(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOBlocks_3D(hitsr, mats, mats_order, mode, tb) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptMatrix * const M = mats[nmodes];
    sptValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];        

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            sptIndex * block_coord = (sptIndex*)malloc(nmodes * sizeof(*block_coord));
            sptIndex * ele_coord = (sptIndex*)malloc(nmodes * sizeof(*ele_coord));
            sptValueVector scratch; // Temporary array
            sptNewValueVector(&scratch, R, R);

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
                // omp_set_lock(&lock);
                for(sptIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
                // omp_unset_lock(&lock);
            }   // End loop entries

            /* Free thread-private space */
            free(block_coord);
            free(ele_coord);
            sptFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    // omp_destroy_lock(&lock);

    return 0;
}


int sptOmpMTTKRPHiCOOBlocks_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb) 
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


    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            sptBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
            sptBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
            sptBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                sptIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                sptIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                sptIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                sptValue entry = vals[z];
                for(sptIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks
    }   // End loop kernels

    return 0;
}



int sptOmpMTTKRPHiCOOBlocks_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOBlocks_3D_MatrixTiling(hitsr, mats, mats_order, mode, tb) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];        

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptNnzIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            sptValue ** blocked_times_mat = (sptValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            sptValueVector scratch; // Temporary array
            sptNewValueVector(&scratch, R, R);

            /* Blocked matrices */
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
                for(sptElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(sptIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    for(sptElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                sptElementIndex const mode_i = hitsr->einds[mode].data[z];
                for(sptElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries

            /* Free thread-private space */
            free(blocked_times_mat);
            sptFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}


int sptOmpMTTKRPHiCOOBlocks_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

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
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            sptValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                sptElementIndex mode_i = hitsr->einds[mode].data[z];
                sptElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                sptElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                sptValue entry = vals[z];

                for(sptElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += entry * 
                        blocked_times_mat_1[(sptBlockMatrixIndex)tmp_i_1 * stride + r] * 
                        blocked_times_mat_2[(sptBlockMatrixIndex)tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}




int sptOmpMTTKRPHiCOOKernelsBlocks(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb) 
{
    omp_set_nested(1);
    omp_set_dynamic(0);

    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOKernelsBlocks_3D(hitsr, mats, mats_order, mode, tk, tb) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptMatrix * const M = mats[nmodes];
    sptValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];        

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            sptIndex * block_coord = (sptIndex*)malloc(nmodes * sizeof(*block_coord));
            sptIndex * ele_coord = (sptIndex*)malloc(nmodes * sizeof(*ele_coord));
            sptValueVector scratch; // Temporary array
            sptNewValueVector(&scratch, R, R);

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
                // omp_set_lock(&lock);
                for(sptIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
                // omp_unset_lock(&lock);
            }   // End loop entries

            /* Free thread-private space */
            free(block_coord);
            free(ele_coord);
            sptFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    // omp_destroy_lock(&lock);

    return 0;
}


int sptOmpMTTKRPHiCOOKernelsBlocks_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb) 
{
    omp_set_nested(1);
    omp_set_dynamic(0);

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


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {

            sptBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
            sptBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
            sptBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                sptIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                sptIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                sptIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                sptValue entry = vals[z];
                for(sptIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks
    }   // End loop kernels

    return 0;
}



int sptOmpMTTKRPHiCOOKernelsBlocks_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb) 
{
    sptIndex const nmodes = hitsr->nmodes;
    omp_set_nested(1);
    omp_set_dynamic(0);

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOOKernelsBlocks_3D_MatrixTiling(hitsr, mats, mats_order, mode, tk, tb) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];        

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptNnzIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            sptValue ** blocked_times_mat = (sptValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            sptValueVector scratch; // Temporary array
            sptNewValueVector(&scratch, R, R);

            /* Blocked matrices */
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
                for(sptElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(sptIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    for(sptElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(sptBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                sptElementIndex const mode_i = hitsr->einds[mode].data[z];
                for(sptElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries

            /* Free thread-private space */
            free(blocked_times_mat);
            sptFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}


int sptOmpMTTKRPHiCOOKernelsBlocks_3D_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptElementIndex const stride = mats[0]->stride;

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
    sptElementIndex const R = mats[mode]->ncols;
    sptRankMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(sptIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        sptNnzIndex kptr_begin = hitsr->kptr.data[k];
        sptNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(sptIndex b=kptr_begin; b<kptr_end; ++b) {
            
            /* Allocate thread-private data */
            sptValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            sptValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            sptNnzIndex bptr_begin = hitsr->bptr.data[b];
            sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
                
                sptElementIndex mode_i = hitsr->einds[mode].data[z];
                sptElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                sptElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                sptValue entry = vals[z];

                for(sptElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(sptBlockMatrixIndex)mode_i * stride + r] += entry * 
                        blocked_times_mat_1[(sptBlockMatrixIndex)tmp_i_1 * stride + r] * 
                        blocked_times_mat_2[(sptBlockMatrixIndex)tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}



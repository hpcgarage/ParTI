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
int sptOmpMTTKRPHiCOOBlocks(
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
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    if(tk > 1 && tb == 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernels(hitsr, mats, mats_order, mode, tk) == 0);
    } else if(tk == 1 && tb > 1) {
        sptAssert(sptOmpMTTKRPHiCOOBlocks(hitsr, mats, mats_order, mode, tb) == 0);
    } else if(tk > 1 && tb > 1) {
        sptAssert(sptOmpMTTKRPHiCOOKernelsBlocks(hitsr, mats, mats_order, mode, tk, tb) == 0);
    } else if(tk == 1 && tb == 1) {
        printf("Should specify sequetial MTTKRP.\n");
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "OMP  HiCOO SpTns MTTKRP");
    sptFreeTimer(timer);

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
    sptNnzIndex const nnz = hitsr->nnz;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const nmats = nmodes - 1;
    sptIndex const stride = mats[0]->stride;
    int result = 0;

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
                sptValue const entry = hitsr->values.data[z];
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


int sptOmpMTTKRPHiCOOBlocks(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tb) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptNnzIndex const nnz = hitsr->nnz;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const nmats = nmodes - 1;
    sptIndex const stride = mats[0]->stride;
    int result = 0;

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
                sptValue const entry = hitsr->values.data[z];
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
    sptNnzIndex const nnz = hitsr->nnz;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const nmats = nmodes - 1;
    sptIndex const stride = mats[0]->stride;
    int result = 0;

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
                sptValue const entry = hitsr->values.data[z];
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


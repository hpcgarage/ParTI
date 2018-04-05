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

#ifndef PARTI_MTTKRP_KERNELS_H
#define PARTI_MTTKRP_KERNELS_H

__device__ void lock(int* mutex);
__device__ void unlock(int* mutex);


/* impl_num = 01 */
__global__ void spt_MTTKRPKernelNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset);


/* impl_num = 02 */
__global__ void spt_MTTKRPKernelNnzRank3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset);


/* impl_num = 03 */
__global__ void spt_MTTKRPKernelNnzRankSplit3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset);


/* impl_num = 04 */
__global__ void spt_MTTKRPKernelRankNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset);


/* impl_num = 05 */
__global__ void spt_MTTKRPKernelRankSplitNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset);


/* impl_num = 06 */
__global__ void spt_MTTKRPKernelRankSplitNnzRB3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptNnzIndex block_offset);


/* impl_num = 09, for arbitraty nmodes. Scratch is necessary for tensors with arbitrary modes. */
__global__ void spt_MTTKRPKernelScratch(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptValue * dev_scratch,
    sptNnzIndex block_offset);



/**** impl_num = 1x: One GPU using one kernel ****/
/* impl_num = 11 */
__global__ void spt_MTTKRPKernelNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);

/* impl_num = 12 */
__global__ void spt_MTTKRPKernelRankNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);

/* impl_num = 15 */
__global__ void spt_MTTKRPKernelRankSplitNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);

/* impl_num = 16 */
__global__ void spt_MTTKRPKernelRankSplitNnzRB3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);



/**** impl_num = 2x: Stream One GPU: cache blocking ****/
/* impl_num = 21. */
__global__ void spt_MTTKRPKernelBlockNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);


/* impl_num = 25 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);


/* impl_num = 26 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnzRB3D(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);



/**** impl_num = 3x: Stream One GPU: shared memory blocking for coarse grain ****/
/* impl_num = 35 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMCoarse(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);

/* impl_num = 36 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMCoarseRB(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);


/**** impl_num = 4x: Stream One GPU: shared memory blocking for medium grain ****/
/* impl_num = 45 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMMedium(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);


/* impl_num = 46 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_SMMediumRB(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex * nnz,
    const sptNnzIndex * dev_nnz_blk_begin,
    const sptIndex R,
    const sptIndex stride,
    sptIndex * const inds_low_allblocks,
    sptIndex ** const inds_low,
    sptIndex ** const Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);


/**** impl_num = 5x: multiple GPUs ****/
/* impl_num = 59, only the interface is a bit different. */
__global__ void spt_MTTKRPKernelScratchDist(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    const sptIndex * inds_low,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptValue * dev_scratch);



/* impl_num = 31 */
__global__ void spt_MTTKRPKernelNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);

__global__ void spt_MTTKRPKernelRankSplitNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);

#endif
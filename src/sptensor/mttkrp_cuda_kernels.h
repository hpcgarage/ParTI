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


/* impl_num = 1 */
__global__ void spt_MTTKRPKernelNnz3D(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    size_t block_offset);


/* impl_num = 2 */
__global__ void spt_MTTKRPKernelNnzRank3D(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    size_t block_offset);


/* impl_num = 3 */
__global__ void spt_MTTKRPKernelNnzRankSplit3D(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    size_t block_offset);


/* impl_num = 4 */
__global__ void spt_MTTKRPKernelRankNnz3D(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    size_t block_offset);


/* impl_num = 5 */
__global__ void spt_MTTKRPKernelRankSplitNnz3D(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    size_t block_offset);



/* impl_num = 09, for arbitraty nmodes. Scratch is necessary for tensors with arbitrary modes. */
__global__ void spt_MTTKRPKernelScratch(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch,
    size_t block_offset);



/**** impl_num = 1x: Stream One GPU ****/
/* impl_num = 11 */
__global__ void spt_MTTKRPKernelBlockNnz3D(
    const size_t mode,
    const size_t nmodes,
    const size_t * nnz,
    const size_t * nnz_blk_begin,
    const size_t R,
    const size_t stride,
    size_t * const inds_low_allblocks,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats);


/* impl_num = 15 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D(
    const size_t mode,
    const size_t nmodes,
    const size_t * nnz,
    const size_t * nnz_blk_begin,
    const size_t R,
    const size_t stride,
    size_t * const inds_low_allblocks,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats);


/* impl_num = 16 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_Coarse(
    const size_t mode,
    const size_t nmodes,
    const size_t * nnz,
    const size_t * dev_nnz_blk_begin,
    const size_t R,
    const size_t stride,
    size_t * const inds_low_allblocks,
    size_t ** const inds_low,
    size_t ** const Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats);


/* impl_num = 17 */
__global__ void spt_MTTKRPKernelBlockRankSplitNnz3D_Medium(
    const size_t mode,
    const size_t nmodes,
    const size_t * nnz,
    const size_t * dev_nnz_blk_begin,
    const size_t R,
    const size_t stride,
    size_t * const inds_low_allblocks,
    size_t ** const inds_low,
    size_t ** const Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats);



/**** impl_num = 2x: multiple GPUs ****/
/* impl_num = 29, only the interface is a bit different. */
__global__ void spt_MTTKRPKernelScratchDist(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    const size_t * inds_low,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch);



#endif
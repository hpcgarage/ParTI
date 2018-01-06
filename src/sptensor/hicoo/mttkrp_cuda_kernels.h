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

/* impl_num = 01  Naive, 1-D */
__global__ void spt_MTTKRPKernelHiCOO_3D_naive(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptElementIndex sc_bits,
    const sptNnzIndex kptr_begin,
    const sptNnzIndex kptr_end,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_cptr,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats);

/* impl_num = 02  Naive, 2-D */
__global__ void spt_MTTKRPKernelRankHiCOO_3D_naive(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptElementIndex sc_bits,
    const sptNnzIndex kptr_begin,
    const sptNnzIndex kptr_end,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_cptr,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats);

/* impl_num = 03  Naive, 2-D */
__global__ void spt_MTTKRPKernelRankSplitHiCOO_3D_naive(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptElementIndex sc_bits,
    const sptNnzIndex kptr_begin,
    const sptNnzIndex kptr_end,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_cptr,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats);

/* impl_num = 04  Naive, 2-D */
__global__ void spt_MTTKRPKernelRankSplitHiCOORB_3D_naive(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptElementIndex sc_bits,
    const sptNnzIndex kptr_begin,
    const sptNnzIndex kptr_end,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_cptr,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats);


#endif

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

#ifndef PARTI_MMUL_KERNELS_H
#define PARTI_MMUL_KERNELS_H

/* impl_num = 01 */
__global__ void spt_TTMNaiveKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, const size_t *X_inds_m,
    const size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset);

/* impl_num = 02 */
__global__ void spt_TTMKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, const size_t *X_inds_m,
    const size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset);



/* impl_num = 11 */
__global__ void spt_TTMNnzKernel(
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride);

/* impl_num = 12 */
__global__ void spt_TTMNnzRankKernel(
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride);

/* impl_num = 13 */
__global__ void spt_TTMRankNnzKernel(
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride);


/* impl_num = 14 */
__global__ void spt_TTMRankRBNnzKernel(
    sptScalar *Y_val, 
    size_t Y_stride, 
    size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride);

/* impl_num = 15 */
__global__ void spt_TTMRankRBNnzKernelSM(
    sptScalar *Y_val, 
    size_t Y_stride, size_t Y_nnz,
    const sptScalar * __restrict__ X_val, 
    size_t X_nnz, 
    const size_t * __restrict__ X_inds_m,
    const size_t * __restrict__ fiberidx_val, 
    size_t fiberidx_len,
    const sptScalar * __restrict__ U_val, 
    size_t U_nrows, 
    size_t U_ncols, 
    size_t U_stride);

#endif
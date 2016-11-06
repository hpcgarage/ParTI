/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

__global__ void spt_TTMNaiveKernel(
    double *Y_val, size_t Y_stride, size_t Y_nnz,
    const double *X_val, size_t X_nnz, const size_t *X_inds_m,
    const size_t *fiberidx_val, size_t fiberidx_len,
    const double *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;

    if(i >= Y_nnz || tidy >= U_ncols) return;
    const size_t inz_begin = fiberidx_val[i];
    const size_t inz_end = fiberidx_val[i+1];

    Y_val[i*Y_stride + tidy] = 0;
    for(size_t j = inz_begin; j < inz_end; ++j) {
        const size_t r = X_inds_m[j];
        Y_val[i*Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
    }
}

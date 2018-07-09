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

#ifndef PTI_ALGORITHM_INCLUDED
#define PTI_ALGORITHM_INCLUDED

#include <vector>
#include <ParTI/device.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>

namespace pti {

SparseTensor tensor_times_matrix(
    SparseTensor& X,
    Tensor& U,
    size_t mode,
    Device* device,
    bool skip_sort = false
);

SparseTensor tensor_times_matrix_omp(
    SparseTensor& X,
    Tensor& U,
    size_t mode,
    bool skip_sort = false
);

SparseTensor tensor_times_matrix_cuda(
    SparseTensor& X,
    Tensor& U,
    size_t mode,
    CudaDevice* cuda_dev,
    bool skip_sort = false
);

void transpose_matrix_inplace(
    Tensor& X,
    bool do_transpose,
    bool want_fortran_style,
    Device *device
);

Tensor unfold(
    SparseTensor& X,
    size_t mode
);

void set_semisparse_indices_by_sparse_ref(
    SparseTensor& dest,
    std::vector<size_t>& fiber_idx,
    SparseTensor& ref,
    size_t mode
);


void set_semisparse_indices_by_sparse_ref_scan_seq(
    SparseTensor& dest,
    std::vector<size_t>& fiber_idx,
    SparseTensor& ref,
    size_t mode
);
void set_semisparse_indices_by_sparse_ref_scan_omp(
    SparseTensor& dest,
    std::vector<size_t>& fiber_idx,
    SparseTensor& ref,
    size_t mode
);
void set_semisparse_indices_by_sparse_ref_scan_omp_task(
    SparseTensor& dest,
    std::vector<size_t>& fiber_idx,
    SparseTensor& ref,
    size_t mode
);
void set_semisparse_indices_by_sparse_ref_scan_cuda(
    SparseTensor& dest,
    std::vector<size_t>& fiber_idx,
    SparseTensor& ref,
    size_t mode
);

void scan_seq(size_t * array, size_t const length);
void scan_omp(size_t * array, size_t const length);
void scan_cuda(size_t * array, size_t const length);


void svd(
    Tensor* U,
    bool U_want_transpose,
    bool U_want_minimal,
    Tensor& S,
    Tensor* V,
    bool V_want_transpose,
    bool V_want_minimal,
    Tensor& X,
    Device* device
);

enum tucker_decomposition_init_type {
    TUCKER_INIT_NVECS,
    TUCKER_INIT_RANDOM,
};

SparseTensor tucker_decomposition(
    SparseTensor& X,
    size_t const R[],
    size_t const dimorder[],
    Device* device,
    enum tucker_decomposition_init_type init = TUCKER_INIT_NVECS,
    double tol = 1.0e-4,
    unsigned maxiters = 50
);

SparseTensor tensor_multiply_scalar(
	SparseTensor& X, 
	Scalar scalar
);

SparseTensor tensor_addition(
	SparseTensor& X,
	SparseTensor& Y
);
}

#endif

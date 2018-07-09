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

#include <ParTI/algorithm.hpp>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/timer.hpp>

namespace pti {

namespace {

void __global__ ttm_cuda_kernel(
    size_t const *__restrict__ fiberidx, size_t const *__restrict__ X_indices_m,
    size_t nrows, size_t ncols, size_t Y_chunk_size, size_t Y_subchunk_size, size_t X_chunk_size, size_t U_stride,
    Scalar *__restrict__ Y_values, Scalar const *__restrict__ X_values, Scalar const *__restrict__ U_values
) {
    size_t i = blockIdx.x;
    size_t inz_begin = fiberidx[i];
    size_t inz_end = fiberidx[i + 1];
    size_t r = threadIdx.x;
    for(size_t k = threadIdx.y; k < Y_subchunk_size; k += blockDim.y) {
        Scalar accumulate = 0;
        for(size_t j = inz_begin; j < inz_end; ++j) {
            size_t c = X_indices_m[j];
            if(r < nrows && c < ncols) {
                accumulate += X_values[j * X_chunk_size + k] * U_values[r * U_stride + c];
            }
        }
        Y_values[i * Y_chunk_size + r * Y_subchunk_size + k] += accumulate;
    }
}

}

SparseTensor tensor_times_matrix_cuda(SparseTensor& X, Tensor& U, size_t mode, CudaDevice* cuda_dev, bool skip_sort) {
    size_t nmodes = X.nmodes;
    size_t nspmodes = X.sparse_order.size();

    ptiCheckError(mode >= nmodes, ERR_SHAPE_MISMATCH, "mode >= X.nmodes");
    ptiCheckError(X.is_dense(cpu)[mode], ERR_UNKNOWN, "X.is_dense[mode] != false");

    ptiCheckError(U.nmodes != 2, ERR_SHAPE_MISMATCH, "U.nmodes != 2");
    ptiCheckError(U.storage_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "U.storage_order[0] != 0");
    ptiCheckError(U.storage_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "U.storage_order[1] != 1");

    size_t nrows = U.shape(cpu)[0];
    size_t ncols = U.shape(cpu)[1];
    size_t Ustride = U.strides(cpu)[1];

    ptiCheckError(X.shape(cpu)[mode] != ncols, ERR_SHAPE_MISMATCH, "X.shape[mode] != U.ncols");

    if(skip_sort) {
        ptiCheckError(X.sparse_order(cpu)[nspmodes - 1] != mode, ERR_SHAPE_MISMATCH, "X.sparse_order[-1] != mode");
    } else {
        Timer timer_sort(cpu);
        timer_sort.start();

        std::unique_ptr<size_t[]> sort_order(new size_t [nspmodes]);
        for(size_t m = 0, i = 0; m < nspmodes; ++m) {
            size_t sort_order_mode = X.sparse_order(cpu)[m];
            if(sort_order_mode != mode) {
                sort_order[i] = sort_order_mode;
                ++i;
            }
        }
        sort_order[nspmodes - 1] = mode;
        X.sort_index(sort_order.get());

        timer_sort.stop();
        timer_sort.print_elapsed_time("CUDA TTM Sort");
    }

    std::unique_ptr<size_t[]> Y_shape(new size_t [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        if(m != mode) {
            Y_shape[m] = X.shape(cpu)[m];
        } else {
            Y_shape[m] = nrows;
        }
    }
    bool const* X_is_dense = X.is_dense(cpu);
    std::unique_ptr<bool[]> Y_is_dense(new bool [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        Y_is_dense[m] = X_is_dense[m] || m == mode;
    }

    SparseTensor Y(nmodes, Y_shape.get(), Y_is_dense.get());
    size_t* X_dense_order = X.dense_order(cpu);
    size_t* Y_dense_order = Y.dense_order(cpu);
    for(size_t m = 0; m < Y.dense_order.size() - 1; ++m) {
        Y_dense_order[m] = X_dense_order[m];
    }
    Y_dense_order[Y.dense_order.size() - 1] = mode;
    Y.sort_index(X.sparse_order(cpu));

    Timer timer_setidx(cpu);
    timer_setidx.start();

    std::vector<size_t> fiberidx;
    set_semisparse_indices_by_sparse_ref(Y, fiberidx, X, mode);

    timer_setidx.stop();
    timer_setidx.print_elapsed_time("CUDA TTM SetIdx");

    Scalar* X_values = X.values(cuda_dev->mem_node);
    Scalar* Y_values = Y.values(cuda_dev->mem_node);
    Scalar* U_values = U.values(cuda_dev->mem_node);
    size_t* X_indices_m = X.indices[mode](cuda_dev->mem_node);
    size_t *dev_fiberidx = (size_t *) session.mem_nodes[cuda_dev->mem_node]->malloc(fiberidx.size() * sizeof (size_t));
    session.mem_nodes[cuda_dev->mem_node]->memcpy_from(dev_fiberidx, fiberidx.data(), *session.mem_nodes[cpu], fiberidx.size() * sizeof (size_t));

    size_t Y_subchunk_size = X.chunk_size;
    size_t Y_num_subchunks = Y.strides(cpu)[mode];
    assert(Y_num_subchunks * Y_subchunk_size == Y.chunk_size);

    Timer timer_kernel(cuda_dev->device_id);
    timer_kernel.start();
    size_t kernel_blockDim_y = std::min(Y_subchunk_size, 1024 / Y_num_subchunks);
    assert(kernel_blockDim_y > 0);
    std::fprintf(stderr, "[CUDA TTM Kernel] Launch ttm_cuda_kernel<<<%zu, (%zu, %zu), 0>>()\n", Y.num_chunks, Y_num_subchunks, kernel_blockDim_y);
    ttm_cuda_kernel<<<Y.num_chunks, dim3(Y_num_subchunks, kernel_blockDim_y), 0>>>(dev_fiberidx, X_indices_m, nrows, ncols, Y.chunk_size, Y_subchunk_size, X.chunk_size, Ustride, Y_values, X_values, U_values);
    int result = cudaThreadSynchronize();
    timer_kernel.stop();
    timer_kernel.print_elapsed_time("CUDA TTM Kernel");
    ptiCheckCUDAError(result != 0);

    session.mem_nodes[cuda_dev->mem_node]->free(dev_fiberidx);

    return Y;
}

}

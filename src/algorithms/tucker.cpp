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
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <ParTI/device.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memblock.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/utils.hpp>

#ifdef PARTI_USE_CUDA
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#endif

namespace pti {

namespace {

void uniform_random_fill_matrix(
    Tensor&   mtx
) {
    ptiCheckError(mtx.nmodes != 2, ERR_SHAPE_MISMATCH, "mtx.nmodes != 2");
    ptiCheckError(mtx.storage_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "mtx.storage_order[0] != 0");
    ptiCheckError(mtx.storage_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "mtx.storage_order[1] != 1");

    std::default_random_engine generator;
    std::uniform_real_distribution<Scalar> distribution(-1.0, 1.0);

    size_t nrows = mtx.shape(cpu)[0];
    size_t ncols = mtx.shape(cpu)[1];
    size_t stride = mtx.strides(cpu)[1];

    Scalar* values = mtx.values(cpu);
    for(size_t i = 0; i < nrows; ++i) {
        for(size_t j = 0; j < ncols; ++j) {
            values[i * stride + j] = distribution(generator);
        }
        for(size_t j = ncols; j < stride; ++j) {
            values[i * stride + j] = 0;
        }
    }
    for(size_t i = nrows * stride; i < mtx.chunk_size; ++i) {
        values[i] = 0;
    }
}

Tensor nvecs(
    SparseTensor& t,
    size_t        n,
    size_t        r,
    Device*       device
) {

    Tensor tm = unfold(t, n);
    Tensor u, s;

    device = session.devices[cpu]; // Experiments show that cuSOLVER is slow when M >> N
    svd(&u, false, true, s, nullptr, false, false, tm, device);
    size_t const* u_shape = u.shape(cpu);
    size_t u_nrows = u_shape[0];
    size_t u_ncols = u_shape[1];
    assert(u_nrows == t.shape(cpu)[n]);

    size_t const result_shape[2] = { u_nrows, r };
    Tensor result(2, result_shape);
    size_t result_m = result_shape[0];
    size_t result_n = result_shape[1];
    size_t result_stride = result.strides(cpu)[1];
    size_t u_stride = u.strides(cpu)[1];

    for(size_t i = 0; i < result_m; ++i) {
        for(size_t j = 0; j < std::min(result_n, u_ncols); ++j) {
            result.values(cpu)[i * result_stride + j] = u.values(cpu)[i * u_stride + j];
        }
    }

    return result;

}

}

SparseTensor tucker_decomposition(
    SparseTensor&   X,
    size_t const    R[],
    size_t const    dimorder[],
    Device*         device,
    enum tucker_decomposition_init_type init,
    double          tol,
    unsigned        maxiters
) {
    maxiters = 2; // jli added for debug
    ptiCheckError(X.dense_order.size() != 0, ERR_SHAPE_MISMATCH, "X should be fully sparse");

    size_t N = X.nmodes;
    double normX = X.norm(device);

    std::unique_ptr<Tensor[]> U(new Tensor[N]);
    size_t U_shape[2];
    for(size_t ni = 1; ni < N; ++ni) {
        size_t n = dimorder[ni];
        U_shape[0] = R[n];
        U_shape[1] = X.shape(cpu)[n];
        U[n].reset(2, U_shape);
        if(false && init == TUCKER_INIT_NVECS) {
            U[n] = nvecs(X, n, R[n], device);
        } else {
            uniform_random_fill_matrix(U[n]);
        }
    }
    SparseTensor core;

    std::unique_ptr<size_t []> sort_order(new size_t [N]);
    std::unique_ptr<SparseTensor []> X_sort_cache(new SparseTensor [N]);
    Timer timer_sort(cpu);
    timer_sort.start();
    for(size_t n = 0; n < N; ++n) {
        Timer timer_sort_i(cpu);
        timer_sort_i.start();
        for(size_t m = 0; m < N; ++m) {
            if(m < n) {
                sort_order[N - m - 1] = m;
            } else if(m != n) {
                sort_order[N - m] = m;
            }
        }
        sort_order[0] = n;
        X_sort_cache[n] = X.clone();
        X_sort_cache[n].sort_index(sort_order.get());
        timer_sort_i.stop();
        timer_sort_i.print_elapsed_time("Tucker Sort");
    }
    timer_sort.stop();
    timer_sort.print_elapsed_time("Tucker Sort Total");


    double fit = 0;
    SparseTensor Utilde_next;
    for(unsigned iter = 0; iter < maxiters; ++iter) {
        Timer timer_iter(cpu);
        timer_iter.start();

        double fitold = fit;

        Timer timer_loop(cpu);
        timer_loop.start();
        SparseTensor* Utilde = &X;
        for(size_t ni = 0; ni < N; ++ni) {
            std::printf("\n");
            size_t n = dimorder[ni];

            Timer timer_ttm_chain(cpu);
            timer_ttm_chain.start();
            Utilde = &X_sort_cache[n];
            for(size_t m = 0; m < N; ++m) {
                if(m != n) {
                    std::printf("[Tucker Decomp]: Iter %u, n = %zu, m = %zu\n", iter, n, m);
                    std::fflush(stdout);
                    Utilde_next = tensor_times_matrix(*Utilde, U[m], m, device, true);
                    Utilde = &Utilde_next;
                }
            }
            timer_ttm_chain.stop();
            timer_ttm_chain.print_elapsed_time("TTM Chain");

            if(device->mem_node != cpu) {
                std::printf("[Tucker TTM Chain]: Releasing GPU memory of X_sort_cache[%zu]\n", n);
                X_sort_cache[n].values.mark_dirty(cpu);
                X_sort_cache[n].values.free(device->mem_node);
                for(size_t m = 0; m < N; ++m) {
                    X_sort_cache[n].indices[m].mark_dirty(cpu);
                    X_sort_cache[n].indices[m].free(device->mem_node);
                }
            }

            Timer timer_svd(device->device_id);
            timer_svd.start();
            // Mode n is sparse, while other modes are dense
            U[n] = nvecs(*Utilde, n, R[n], device);
            timer_svd.stop();
            timer_svd.print_elapsed_time("SVD");

            transpose_matrix_inplace(U[n], true, false, device);
        }   // End loop of nmodes
        timer_loop.stop();
        timer_loop.print_elapsed_time("Tucker Decomp Loop");

        std::printf("\n");
        std::fflush(stdout);

        Timer timer_core(cpu);
        timer_core.start();
        core = tensor_times_matrix(*Utilde, U[dimorder[N-1]], dimorder[N-1], device, true);
        timer_core.stop();
        timer_core.print_elapsed_time("Tucker Decomp Core");

        Timer timer_fit(cpu);
        timer_fit.start();
        double normCore = core.norm(device);
        double normResidual = std::sqrt(normX * normX - normCore * normCore);
        fit = 1 - normResidual / normX;
        double fitchange = std::fabs(fitold - fit);
        timer_fit.stop();
        timer_fit.print_elapsed_time("Tucker Decomp Norm");

        std::printf("[Tucker Dcomp]: normX = %lg, normCore = %lg\n", normX, normCore);
        std::printf("[Tucker Dcomp]: fit = %lg, fitchange = %lg\n", fit, fitchange);
        std::fflush(stdout);

        if(iter != 0 && fitchange < tol) {
            break;
        }

        timer_iter.stop();
        timer_iter.print_elapsed_time("Tucker Decomp Iter");
    }   // End of iterations

    return core;
}

}

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
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memblock.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/utils.hpp>

namespace pti {

Tensor unfold(
    SparseTensor& X,
    size_t mode
) {
    ptiCheckError(X.is_dense(cpu)[mode] != false, ERR_SHAPE_MISMATCH, "X.is_dense[mode] != false");
    ptiCheckError(X.sparse_order.size() != 1, ERR_SHAPE_MISMATCH, "X.sparse_order.size() != 1")

    size_t const* shape = X.shape(cpu);
    size_t nrows = shape[mode];
    size_t ncols = 1;
    for(size_t m = 0; m < X.nmodes; ++m) {
        if(m != mode) {
            ncols *= shape[m];
        }
    }
    size_t const result_shape[2] = { nrows, ncols };

    Tensor result(2, result_shape);
    size_t result_stride = result.strides(cpu)[1];

    std::unique_ptr<size_t[]> coordinate(new size_t [X.nmodes]);

    for(size_t i = 0; i < X.num_chunks; ++i) {
        size_t row = X.indices[mode](cpu)[i];
        size_t col = 0;
        std::memset(coordinate.get(), 0, X.nmodes * sizeof (size_t));
        size_t first_mode = mode == 0 ? 1 : 0;
        size_t last_mode = mode == X.nmodes - 1 ? X.nmodes - 2 : X.nmodes - 1;
        assert(last_mode >= first_mode);
        while(coordinate[first_mode] < shape[first_mode]) {
            size_t j = X.indices_to_intra_offset(coordinate.get());
            result.values(cpu)[row * result_stride + col] = X.values(cpu)[i * X.chunk_size + j];

            ++col;
            ++coordinate[last_mode];
            for(size_t m = last_mode; m != first_mode; --m) {
                if(mode == m) {
                    continue;
                } else if(coordinate[m] >= shape[m]) {
                    coordinate[m] = 0;
                    ++coordinate[(mode == m - 1) ? m - 2 : m - 1];
                } else {
                    break;
                }
            }
        }
    }

    return result;
}


}

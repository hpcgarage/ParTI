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

#include <ParTI/sptensor.hpp>
#include <cstring>
#include <memory>
#include <utility>
#include <ParTI/utils.hpp>

namespace pti {

SparseTensor SparseTensor::to_fully_dense() {
    std::unique_ptr<bool[]> mode_is_dense(new bool [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        mode_is_dense[m] = true;
    }

    SparseTensor result(nmodes, shape(cpu), mode_is_dense.get());

    size_t* result_dense_order = result.dense_order(cpu);
    std::memcpy(result_dense_order, sparse_order(cpu), sparse_order.size() * sizeof (size_t));
    std::memcpy(result_dense_order + sparse_order.size(), dense_order(cpu), dense_order.size() * sizeof (size_t));

    result.init_single_chunk();

    std::unique_ptr<size_t[]> coord(new size_t [nmodes]);
    for(size_t i = 0; i < num_chunks; ++i) {
        for(size_t j = i * chunk_size; j < i * chunk_size + chunk_size; ++j) {
            bool inbound = offset_to_indices(coord.get(), j);
            if(inbound) {
                size_t offset = 0;
                if(nmodes != 0) {
                    for(size_t m = 0; m + 1 < nmodes; ++m) {
                        offset += coord[result_dense_order[m]];
                        offset *= result.strides(cpu)[result_dense_order[m + 1]];
                    }
                    offset += coord[result_dense_order[nmodes - 1]];
                }
                result.values(cpu)[offset] = values(cpu)[j];
                //std::fprintf(stderr, "(%s) => %zu, value = %f\n", array_to_string(coord.get(), nmodes).c_str(), offset, values(cpu)[j]);
            }
        }
    }

    result.num_chunks = 1;

    return result;
}

}

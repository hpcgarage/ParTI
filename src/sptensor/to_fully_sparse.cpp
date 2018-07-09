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

namespace pti {

SparseTensor SparseTensor::to_fully_sparse() {
    std::unique_ptr<bool[]> mode_is_dense(new bool [nmodes] ());

    SparseTensor result(nmodes, shape(cpu), mode_is_dense.get());

    std::memcpy(result.sparse_order(cpu), sparse_order(cpu), sparse_order.size() * sizeof (size_t));
    std::memcpy(result.sparse_order(cpu) + sparse_order.size(), dense_order(cpu), dense_order.size() * sizeof (size_t));

    result.reserve(num_chunks * chunk_size, false);

    std::unique_ptr<size_t[]> coord(new size_t [nmodes]);
    for(size_t i = 0; i < num_chunks; ++i) {
        for(size_t j = i * chunk_size; j < i * chunk_size + chunk_size; ++j) {
            bool inbound = offset_to_indices(coord.get(), j);
            if(inbound) {
                result.append(coord.get(), values(cpu)[j]);
            }
        }
    }

    return result;
}

}

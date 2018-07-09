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
#include <ParTI/error.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/utils.hpp>

namespace pti {

void SparseTensor::put(size_t const location, size_t const coord[], Scalar const value[]) {

    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense(cpu)[m]) {
            indices[m](cpu)[location] = coord[m];
        }
    }
    std::memcpy(&values(cpu)[location * chunk_size], value, chunk_size * sizeof (Scalar));
}

void SparseTensor::put(size_t const location, size_t const coord[], Scalar value) {
    ptiCheckError(chunk_size != 1, ERR_SHAPE_MISMATCH, "tensor is not fully sparse");

    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense(cpu)[m]) {
            indices[m](cpu)[location] = coord[m];
        }
    }
    values(cpu)[location * chunk_size] = value;
    ++num_chunks;
}

}

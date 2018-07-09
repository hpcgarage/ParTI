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

void SparseTensor::append(size_t const coord[], Scalar const value[]) {
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense(cpu)[m]) {
            if(indices[m].size() < num_chunks + 1) { // Need reallocation
                size_t new_size = indices[m].size() >= 8 ?
                    indices[m].size() + indices[m].size() / 2 :
                    8;
                if(new_size < num_chunks + 1) {
                    new_size = num_chunks + 1;
                }
                indices[m].resize(cpu, new_size);
            }
        }
    }

    if(values.size() < (num_chunks + 1) * chunk_size) { // Need reallocation
        size_t new_size = values.size() >= 8 ?
            values.size() + values.size() / 2 :
            8;
        if(new_size < (num_chunks + 1) * chunk_size) {
            new_size = (num_chunks + 1) * chunk_size;
        }
        values.resize(cpu, new_size);
    }

    size_t next_offset = num_chunks;
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense(cpu)[m]) {
            indices[m](cpu)[next_offset] = coord[m];
        }
    }
    std::memcpy(&values(cpu)[next_offset * chunk_size], value, chunk_size * sizeof (Scalar));
    ++num_chunks;
}

void SparseTensor::append(size_t const coord[], Scalar value) {
    ptiCheckError(chunk_size != 1, ERR_SHAPE_MISMATCH, "tensor is not fully sparse");

    for(size_t m = 0; m < nmodes; ++m) {
        if(indices[m].size() < num_chunks + 1) { // Need reallocation
            size_t new_size = indices[m].size() >= 8 ?
                indices[m].size() + indices[m].size() / 2 :
                8;
            if(new_size < num_chunks + 1) {
                new_size = num_chunks + 1;
            }
            indices[m].resize(cpu, new_size);
        }
    }

    if(values.size() < (num_chunks + 1) * chunk_size) { // Need reallocation
        size_t new_size = values.size() >= 8 ?
            values.size() + values.size() / 2 :
            8;
        if(new_size < (num_chunks + 1) * chunk_size) {
            new_size = (num_chunks + 1) * chunk_size;
        }
        values.resize(cpu, new_size);
    }

    size_t next_offset = num_chunks;
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense(cpu)[m]) {
            indices[m](cpu)[next_offset] = coord[m];
        }
    }
    values(cpu)[next_offset * chunk_size] = value;
    ++num_chunks;
}

}

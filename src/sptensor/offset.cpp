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
#include <ParTI/utils.hpp>

namespace pti {

bool SparseTensor::offset_to_indices(size_t indices[], size_t offset) {
    if(offset >= num_chunks * chunk_size) {
        std::memcpy(indices, this->shape(cpu), nmodes * sizeof (size_t));
        return false;
    }
    bool* is_dense = this->is_dense(cpu);
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense[m]) {
            indices[m] = this->indices[m](cpu)[offset / chunk_size];
        }
    }
    if(chunk_size != 1) { // Semi sparse tensor
        size_t intra_chunk = offset % chunk_size;
        size_t* dense_order = this->dense_order(cpu);
        size_t* strides = this->strides(cpu);
        for(size_t o = this->dense_order.size()-1; o != 0; --o) {
            size_t m = dense_order[o];
            indices[m] = intra_chunk % strides[m];
            intra_chunk /= strides[m];
        }
        indices[dense_order[0]] = intra_chunk;
    }
    bool inbound = true;
    for(size_t m = 0; m < this->nmodes; ++m) {
        inbound = inbound && indices[m] < this->shape(cpu)[m];
    }
    return inbound;
}

size_t SparseTensor::indices_to_intra_offset(size_t const indices[]) {
    size_t const* dense_order = this->dense_order(cpu);
    size_t const* strides = this->strides(cpu);
    size_t offset = indices[dense_order[0]];
    for(size_t o = 1; o < this->dense_order.size(); ++o) {
        size_t m = dense_order[o];
        offset *= strides[m];
        offset += indices[m];
    }

    return offset;
}

}

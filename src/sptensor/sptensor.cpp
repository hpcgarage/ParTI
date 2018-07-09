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
#include <utility>
#include <ParTI/utils.hpp>

namespace pti {

SparseTensor::SparseTensor() {
    this->nmodes = 0;
    this->chunk_size = 0;
    this->num_chunks = 0;
    this->indices = nullptr;
}

SparseTensor::SparseTensor(size_t nmodes, size_t const shape[], bool const is_dense[]) {
    this->indices = nullptr;
    reset(nmodes, shape, is_dense);
}

SparseTensor::SparseTensor(SparseTensor&& other) {
    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->is_dense = std::move(other.is_dense);
    this->dense_order = std::move(other.dense_order);
    this->sparse_order = std::move(other.sparse_order);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->num_chunks = other.num_chunks;
    this->indices = other.indices;
    this->values = std::move(other.values);

    other.nmodes = 0;
    other.chunk_size = 0;
    other.num_chunks = 0;
    other.indices = nullptr;
}

SparseTensor& SparseTensor::operator= (SparseTensor&& other) {
    delete[] this->indices;

    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->is_dense = std::move(other.is_dense);
    this->dense_order = std::move(other.dense_order);
    this->sparse_order = std::move(other.sparse_order);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->num_chunks = other.num_chunks;
    this->indices = other.indices;
    this->values = std::move(other.values);

    other.nmodes = 0;
    other.chunk_size = 0;
    other.num_chunks = 0;
    other.indices = nullptr;

    return *this;
}

SparseTensor SparseTensor::clone() {
    SparseTensor result;

    // nmodes
    result.nmodes = nmodes;

    // shape
    result.shape = shape.clone(cpu);

    // is_dense
    result.is_dense = is_dense.clone(cpu);

    // dense_order
    result.dense_order = dense_order.clone(cpu);

    // sparse_order
    result.sparse_order = sparse_order.clone(cpu);

    // strides
    result.strides = strides.clone(cpu);

    // chunk_size
    result.chunk_size = chunk_size;

    // num_chunks
    result.num_chunks = num_chunks;

    // indices
    result.indices = new MemBlock<size_t[]> [nmodes];
    for(size_t m = 0; m < nmodes; ++m) {
        result.indices[m] = indices[m].clone(cpu);
    }

    // values
    result.values = values.clone(cpu);

    return result;
}

SparseTensor& SparseTensor::reset(size_t nmodes, size_t const shape[], bool const is_dense[]) {

    // nmodes
    this->nmodes = nmodes;

    // shape
    this->shape.allocate(cpu, nmodes);
    std::memcpy(this->shape(cpu), shape, nmodes * sizeof (size_t));

    // is_dense
    this->is_dense.allocate(cpu, nmodes);
    std::memcpy(this->is_dense(cpu), is_dense, nmodes * sizeof (bool));

    size_t dense_modes = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        if(is_dense[m]) {
            ++dense_modes;
        }
    }

    // dense_order
    this->dense_order.allocate(cpu, dense_modes);
    size_t* dense_order = this->dense_order(cpu);
    size_t order_idx = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        if(is_dense[m]) {
            dense_order[order_idx++] = m;
        }
    }

    // sparse_order
    this->sparse_order.allocate(cpu, nmodes - dense_modes);
    size_t* sparse_order = this->sparse_order(cpu);
    order_idx = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense[m]) {
            sparse_order[order_idx++] = m;
        }
    }

    // strides
    this->strides.allocate(cpu, nmodes);
    size_t* strides = this->strides(cpu);
    for(size_t m = 0; m < nmodes; ++m) {
        if(is_dense[m]) {
            strides[m] = ceil_div<size_t>(shape[m], 8) * 8;
        } else {
            strides[m] = 1;
        }
    }

    // chunk_size
    this->chunk_size = 1;
    for(size_t m = 0; m < nmodes; ++m) {
        this->chunk_size *= strides[m];
    }

    // num_chunks
    this->num_chunks = 0;

    // indices
    delete[] this->indices;
    this->indices = new MemBlock<size_t[]> [nmodes];

    // values
    this->values.allocate(cpu, 0);

    return *this;
}

SparseTensor::~SparseTensor() {
    delete[] this->indices;
}

}

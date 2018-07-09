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

#include <ParTI/tensor.hpp>
#include <cstring>
#include <utility>
#include <ParTI/utils.hpp>

namespace pti {

Tensor::Tensor() {
    this->nmodes = 0;
    this->chunk_size = 0;
}

Tensor::Tensor(size_t nmodes, size_t const shape[], bool initialize) {
    reset(nmodes, shape, initialize);
}

Tensor::Tensor(Tensor&& other) {
    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->storage_order = std::move(other.storage_order);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->values = std::move(other.values);

    other.nmodes = 0;
    other.chunk_size = 0;
}

Tensor& Tensor::operator= (Tensor&& other) {
    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->storage_order = std::move(other.storage_order);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->values = std::move(other.values);

    other.nmodes = 0;
    other.chunk_size = 0;

    return *this;
}

Tensor& Tensor::reset(size_t nmodes, size_t const shape[], bool initialize) {

    // nmodes
    this->nmodes = nmodes;

    // shape
    this->shape.allocate(cpu, nmodes);
    std::memcpy(this->shape(cpu), shape, nmodes * sizeof (size_t));

    // storage_order
    this->storage_order.allocate(cpu, nmodes);
    size_t* storage_order = this->storage_order(cpu);
    size_t order_idx = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        storage_order[order_idx++] = m;
    }

    // strides
    this->strides.allocate(cpu, nmodes);
    size_t* strides = this->strides(cpu);
    for(size_t m = 0; m < nmodes; ++m) {
        strides[m] = ceil_div<size_t>(shape[m], 8) * 8;
    }

    // chunk_size
    this->chunk_size = 1;
    for(size_t m = 0; m < nmodes; ++m) {
        this->chunk_size *= strides[m];
    }

    // values
    this->values.allocate(cpu, chunk_size);
    if(initialize) {
        std::memset(this->values(cpu), 0, chunk_size * sizeof (Scalar));
    }

    return *this;
}

Tensor::~Tensor() {
}

}

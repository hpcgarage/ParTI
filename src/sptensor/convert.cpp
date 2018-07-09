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
#include <utility>
#include <ParTI/tensor.hpp>
#include <ParTI/utils.hpp>

namespace pti {

SparseTensor::SparseTensor(Tensor&& other) {
    delete[] this->indices;

    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->is_dense.allocate(cpu, nmodes);
    this->dense_order = std::move(other.storage_order);
    this->sparse_order.allocate(cpu, 0);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->num_chunks = 1;
    this->indices = new MemBlock<size_t[]> [nmodes];
    this->values = std::move(other.values);

    bool* is_dense = this->is_dense(cpu);
    for(size_t m = 0; m < nmodes; ++m) {
        is_dense[m] = true;
    }

    for(size_t m = 0; m < nmodes; ++m) {
        this->indices[m].allocate(cpu, 0);
        this->indices[m].ptr(cpu)[0] = 0;
    }

    other.nmodes = 0;
    other.chunk_size = 0;
}

SparseTensor& SparseTensor::operator= (Tensor&& other) {
    delete[] this->indices;

    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->is_dense.allocate(cpu, nmodes);
    this->dense_order = std::move(other.storage_order);
    this->sparse_order.allocate(cpu, 0);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->num_chunks = 1;
    this->indices = new MemBlock<size_t[]> [nmodes];
    this->values = std::move(other.values);

    bool* is_dense = this->is_dense(cpu);
    for(size_t m = 0; m < nmodes; ++m) {
        is_dense[m] = true;
    }

    for(size_t m = 0; m < nmodes; ++m) {
        this->indices[m].allocate(cpu, 0);
        this->indices[m].ptr(cpu)[0] = 0;
    }

    other.nmodes = 0;
    other.chunk_size = 0;

    return *this;
}

}

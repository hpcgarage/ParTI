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
#include <utility>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/utils.hpp>

namespace pti {

Tensor::Tensor(SparseTensor&& other) {
    bool const* is_dense = other.is_dense(cpu);
    for(size_t m = 0; m < other.nmodes; ++m) {
        ptiCheckError(!is_dense[m], ERR_SHAPE_MISMATCH, "tensor is not fully dense");
    }

    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->storage_order = std::move(other.dense_order);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->values = std::move(other.values);

    other.nmodes = 0;
    other.chunk_size = 0;
    delete[] other.indices;
    other.indices = nullptr;
}

Tensor& Tensor::operator= (SparseTensor&& other) {
    bool const* is_dense = other.is_dense(cpu);
    for(size_t m = 0; m < other.nmodes; ++m) {
        ptiCheckError(!is_dense[m], ERR_SHAPE_MISMATCH, "tensor is not fully dense");
    }

    this->nmodes = other.nmodes;
    this->shape = std::move(other.shape);
    this->storage_order = std::move(other.dense_order);
    this->strides = std::move(other.strides);
    this->chunk_size = other.chunk_size;
    this->values = std::move(other.values);

    other.nmodes = 0;
    other.chunk_size = 0;
    delete[] other.indices;
    other.indices = nullptr;

    return *this;
}

}

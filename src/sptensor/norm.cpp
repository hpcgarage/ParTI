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
#include <cmath>
#include <cstring>
#include <memory>

namespace pti {

namespace {

double sqnorm_fully_sparse(SparseTensor &tensor, Device *device) {
    (void) device;

    double sqnorm = 0;
    const Scalar *values = tensor.values(cpu);

    for(size_t i = 0; i < tensor.num_chunks; ++i) {
        double cell_value = values[i * tensor.chunk_size];
        sqnorm += cell_value * cell_value;
    }

    return sqnorm;
}

double sqnorm_fully_dense_1d(SparseTensor &tensor, Device *device) {
    (void) device;

    double sqnorm = 0;
    const size_t *shape = tensor.shape(cpu);
    const Scalar *values = tensor.values(cpu);

    for(size_t x = 0; x < shape[0]; ++x) {
        double cell_value = values[x];
        sqnorm += cell_value * cell_value;
    }
    return sqnorm;
}

double sqnorm_fully_dense_2d(SparseTensor &tensor, Device *device) {
    (void) device;

    double sqnorm = 0;
    const size_t *shape = tensor.shape(cpu);
    const size_t *strides = tensor.strides(cpu);
    const size_t *dense_order = tensor.dense_order(cpu);
    const Scalar *values = tensor.values(cpu);

    for(size_t x = 0; x < shape[dense_order[0]]; ++x) {
        for(size_t y = 0; y < shape[dense_order[1]]; ++y) {
            double cell_value = values[x * strides[dense_order[1]] + y];
            sqnorm += cell_value * cell_value;
        }
    }
    return sqnorm;
}

double sqnorm_fully_dense_3d(SparseTensor &tensor, Device *device) {
    (void) device;

    double sqnorm = 0;
    const size_t *shape = tensor.shape(cpu);
    const size_t *strides = tensor.strides(cpu);
    const size_t *dense_order = tensor.dense_order(cpu);
    const Scalar *values = tensor.values(cpu);

    for(size_t x = 0; x < shape[dense_order[0]]; ++x) {
        for(size_t y = 0; y < shape[dense_order[1]]; ++y) {
            for(size_t z = 0; z < shape[dense_order[2]]; ++z) {
                double cell_value = values[
                    x * strides[dense_order[1]] * strides[dense_order[2]] +
                    y * strides[dense_order[2]] +
                    z
                ];
                sqnorm += cell_value * cell_value;
            }
        }
    }
    return sqnorm;
}

double sqnorm_fully_dense_4d(SparseTensor &tensor, Device *device) {
    (void) device;

    double sqnorm = 0;
    const size_t *shape = tensor.shape(cpu);
    const size_t *strides = tensor.strides(cpu);
    const size_t *dense_order = tensor.dense_order(cpu);
    const Scalar *values = tensor.values(cpu);

    for(size_t w = 0; w < shape[dense_order[0]]; ++w) {
        for(size_t x = 0; x < shape[dense_order[1]]; ++x) {
            for(size_t y = 0; y < shape[dense_order[2]]; ++y) {
                for(size_t z = 0; z < shape[dense_order[3]]; ++z) {
                    double cell_value = values[
                        w * strides[dense_order[1]] * strides[dense_order[2]] * strides[dense_order[3]] +
                        x * strides[dense_order[2]] * strides[dense_order[3]] +
                        y * strides[dense_order[3]] +
                        z
                    ];
                    sqnorm += cell_value * cell_value;
                }
            }
        }
    }
    return sqnorm;
}

}

double SparseTensor::norm(Device *device) {
    size_t num_dense_order = this->dense_order.size();

    if(num_dense_order == 0) {
        return std::sqrt(sqnorm_fully_sparse(*this, device));
    }
    if(num_dense_order == nmodes) {
        if(num_dense_order == 1) {
            return std::sqrt(sqnorm_fully_dense_1d(*this, device));
        }
        if(num_dense_order == 2) {
            return std::sqrt(sqnorm_fully_dense_2d(*this, device));
        }
        if(num_dense_order == 3) {
            return std::sqrt(sqnorm_fully_dense_3d(*this, device));
        }
        if(num_dense_order == 4) {
            return std::sqrt(sqnorm_fully_dense_4d(*this, device));
        }
    }

    double sqnorm = 0;
    const size_t *shape = this->shape(cpu);
    const size_t *dense_order = this->dense_order(cpu);
    std::unique_ptr<size_t []> coord(new size_t [num_dense_order]);
    const Scalar *values = this->values(cpu);

    for(size_t i = 0; i < num_chunks; ++i) {
        std::memset(coord.get(), 0, nmodes * sizeof (size_t));
        for(;;) {
            for(size_t m = num_dense_order - 1; m != 0; --m) {
                if(coord[m] >= shape[dense_order[m]]) {
                    coord[m] = 0;
                    ++coord[m - 1];
                } else {
                    break;
                }
            }
            if(coord[0] >= shape[dense_order[0]]) {
                break;
            }
            double cell_value = values[i * chunk_size + indices_to_intra_offset(coord.get())];
            sqnorm += cell_value * cell_value;
            ++coord[num_dense_order - 1];
        }
    }

    return std::sqrt(sqnorm);
}

}

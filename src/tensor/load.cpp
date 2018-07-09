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
#include <cstdio>
#include <cstring>
#include <memory>
#include <ParTI/error.hpp>

namespace pti {

Tensor Tensor::load(std::FILE* fp) {
    int io_result;

    size_t nmodes;
    io_result = std::fscanf(fp, "%zu", &nmodes);
    ptiCheckOSError(io_result != 1);

    std::unique_ptr<size_t[]> coordinate(new size_t [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        io_result = std::fscanf(fp, "%zu", &coordinate[m]);
        ptiCheckOSError(io_result != 1);
    }

    Tensor tensor(nmodes, coordinate.get());

    if(tensor.chunk_size == 0) {
        return tensor;
    }

    std::memset(coordinate.get(), 0, nmodes * sizeof (size_t));
    size_t const* shape = tensor.shape(cpu);
    Scalar* values = tensor.values(cpu);
    while(coordinate[0] < shape[0]) {
        double value;
        io_result = std::fscanf(fp, "%lg", &value);
        ptiCheckOSError(io_result < 0);

        size_t offset = tensor.indices_to_offset(coordinate.get());
        values[offset] = value;

        ++coordinate[nmodes - 1];
        for(size_t m = nmodes - 1; m != 0; --m) {
            if(coordinate[m] >= shape[m]) {
                coordinate[m] = 0;
                ++coordinate[m - 1];
            } else {
                break;
            }
        }
    }

    return tensor;
}

}

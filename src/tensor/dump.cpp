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
#include <memory>
#include <ParTI/error.hpp>
#include <ParTI/utils.hpp>

namespace pti {

void Tensor::dump(std::FILE* fp) {
    int io_result;

    io_result = std::fprintf(fp, "%zu\n", nmodes);
    ptiCheckOSError(io_result < 0);

    io_result = std::fprintf(fp, "%s\n\n", array_to_string(shape(cpu), shape.size(), "\t").c_str());
    ptiCheckOSError(io_result < 0);

    if(chunk_size == 0) {
        return;
    }

    std::unique_ptr<size_t[]> coordinate(new size_t [nmodes]());
    size_t const* shape = this->shape(cpu);
    Scalar const* values = this->values(cpu);
    while(coordinate[0] < shape[0]) {
        size_t offset = this->indices_to_offset(coordinate.get());
        std::fprintf(fp, "% .16lg", (double) values[offset]);
        ptiCheckOSError(io_result < 0);

        ++coordinate[nmodes - 1];
        for(size_t m = nmodes - 1; m != 0; --m) {
            if(coordinate[m] >= shape[m]) {
                std::fprintf(fp, "\n");
                ptiCheckOSError(io_result < 0);
                coordinate[m] = 0;
                ++coordinate[m - 1];
            } else {
                if(m == nmodes - 1) {
                    std::fprintf(fp, "\t");
                    ptiCheckOSError(io_result < 0);
                }
                break;
            }
        }
    }
}

}

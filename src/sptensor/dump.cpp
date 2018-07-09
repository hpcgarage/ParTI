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
#include <cstdio>
#include <memory>
#include <ParTI/error.hpp>
#include <ParTI/utils.hpp>

namespace pti {

void SparseTensor::dump(std::FILE* fp, size_t start_index) {
    int io_result;

    io_result = std::fprintf(fp, "%zu\n", nmodes);
    ptiCheckOSError(io_result < 0);

    io_result = std::fprintf(fp, "%s\n", array_to_string(shape(cpu), shape.size(), "\t").c_str());
    ptiCheckOSError(io_result < 0);

    std::unique_ptr<size_t[]> coordinate(new size_t [nmodes]);
    Scalar const* values = this->values(cpu);
    for(size_t i = 0; i < num_chunks * chunk_size; ++i) {
        bool inbound = offset_to_indices(coordinate.get(), i);
        if(inbound) {
            for(size_t m = 0; m < nmodes; ++m) {
                coordinate[m] += start_index;
            }
            io_result = std::fprintf(fp, "%s\t% .16lg\n",
                array_to_string(coordinate.get(), nmodes, "\t").c_str(),
                (double) values[i]);
            ptiCheckOSError(io_result < 0);
        }
    }
}

}

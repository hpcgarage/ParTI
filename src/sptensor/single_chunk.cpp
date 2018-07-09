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
#include <algorithm>
#include <cstring>

namespace pti {

void SparseTensor::init_single_chunk(bool initialize) {
    for(size_t m = 0; m < nmodes; ++m) {
        if(indices[m].size() != 1) { // Need reallocation
            indices[m].allocate(cpu, 1);
        }
        indices[m](cpu)[0] = 0;
    }

    if(values.size() != chunk_size) { // Need reallocation
        values.allocate(cpu, chunk_size);
        if(initialize) {
            std::memset(values(cpu), 0, chunk_size * sizeof (Scalar));
        }
    }

    num_chunks = 1;
}

}

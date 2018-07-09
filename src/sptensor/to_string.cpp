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
#include <memory>
#include <string>
#include <ParTI/utils.hpp>

namespace pti {

namespace {

int compare_indices(size_t const i[], size_t const j[], size_t const mode_order[], size_t nmodes) {
    for(size_t m = 0; m < nmodes; ++m) {
        size_t mode = mode_order[m];
        size_t idx_i = i[mode];
        size_t idx_j = j[mode];
        if(idx_i < idx_j) {
            return -1;
        } else if(idx_i > idx_j) {
            return 1;
        }
    }
    return 0;
}

}

std::string SparseTensor::to_string(bool sparse_format, size_t limit) {
    std::string result = "pti::SparseTensor(\n  shape = [";
    result += array_to_string(shape(cpu), nmodes);
    result += "], strides = [";
    result += array_to_string(strides(cpu), strides.size());
    result += "],\n  is_dense = [";
    result += array_to_string(is_dense(cpu), is_dense.size());
    result += "],\n  dense_order = [";
    result += array_to_string(dense_order(cpu), dense_order.size());
    result += "], sparse_order = [";
    result += array_to_string(sparse_order(cpu), sparse_order.size());
    result += "],\n  values[";
    result += std::to_string(num_chunks);
    result += 'x';
    result += std::to_string(chunk_size);
    result += "] = {\n";
    if(sparse_format) {
        for(size_t i = 0; i < num_chunks; ++i) {
            if(limit != 0 && i >= limit) {
                result += ",\n    ...";
                break;
            }
            if(i != 0) {
                result += ",\n";
            }
            result += "    (";
            for(size_t m = 0; m < nmodes; ++m) {
                if(m != 0) {
                    result += ", ";
                }
                if(is_dense(cpu)[m]) {
                    result += ':';
                } else {
                    result += std::to_string(indices[m](cpu)[i]);
                }
            }
            result += "): [";
            for(size_t j = 0; j < chunk_size; ++j) {
                if(limit != 0 && j >= limit) {
                    result += ", ...";
                    break;
                }
                if(j != 0) {
                    result += ", ";
                }
                Scalar value = values(cpu)[i * chunk_size + j];
                if(value >= 0) {
                    result += ' ';
                }
                result += std::to_string(value);
            }
            result += ']';
        }
        if(num_chunks != 0) {
            result += '\n';
        }
    } else if(nmodes != 0) {
        // mode_order = concat(sparse_order, dense_order)
        std::unique_ptr<size_t[]> mode_order(new size_t [nmodes]);
        std::memcpy(mode_order.get(), sparse_order(cpu), sparse_order.size() * sizeof (size_t));
        std::memcpy(mode_order.get() + sparse_order.size(), dense_order(cpu), dense_order.size() * sizeof (size_t));

        size_t nonzero_modes = 0;
        for(size_t m = 0; m < nmodes; ++m) {
            if(shape(cpu)[mode_order[m]] == 0) {
                break;
            }
            ++nonzero_modes;
        }
        if(nonzero_modes != nmodes) {
            ++nonzero_modes;
        }

        size_t i = 0;
        size_t level = 0;
        size_t first_in_level = false;

        std::unique_ptr<size_t[]> coord(new size_t [nmodes] ());
        std::unique_ptr<size_t[]> next_coord(new size_t [nmodes]);
        bool inbound = offset_to_indices(next_coord.get(), i);

        do {

            // Ascending levels
            if(level != nonzero_modes) {
                if(level != 0) {
                    result += ",\n";
                }
                for(size_t m = 0; m < level + 4; ++m) {
                    result += ' ';
                }
                for(size_t m = level; m < nonzero_modes; ++m) {
                    result += '[';
                }
                level = nonzero_modes;
                first_in_level = true;
            }
            size_t mode = mode_order[level - 1];

            // Compare current non-zero element to current coordinate
            int coord_compare = compare_indices(next_coord.get(), coord.get(), mode_order.get(), nmodes);

            if(inbound) {
                if(coord_compare >= 0) {
                    if(first_in_level) {
                        first_in_level = false;
                    } else {
                        result += ", ";
                    }
                    if(coord_compare == 0) {
                        // Print out current element
                        Scalar value = values(cpu)[i];
                        if(value >= 0) {
                            result += ' ';
                        }
                        result += std::to_string(value);
                        ++i;
                        offset_to_indices(next_coord.get(), i);
                    } else {
                        // Print out placeholder
                        result += " 0.000000";
                    }
                    ++coord[mode];
                } else {
                    // Skip out-of-bounds
                    ++i;
                    offset_to_indices(next_coord.get(), i);
                    continue;
                }
            }

            // Calculate carry
            while(level != 0) {
                if(limit != 0 && coord[mode] >= limit) {
                    result += ", ...";
                } else if(coord[mode] < shape(cpu)[mode]) {
                    break;
                }
                coord[mode] = 0;
                --level;
                result += ']';
                if(level == 0) {
                    break;
                }
                mode = mode_order[level - 1];
                ++coord[mode];
            }

        } while(level != 0);
        result += "\n";
    }
    result += "  }\n)";
    return result;
}

}

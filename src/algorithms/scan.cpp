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

#include <ParTI/algorithm.hpp>
#include <cassert>
#include <memory>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/utils.hpp>
#include <omp.h>

namespace pti {

namespace {

static size_t prefix_sum_omp_task( size_t * array, size_t begin, size_t end, size_t const base_length) {
    size_t length  = end - begin;
    size_t mid     = begin + length/2;
    size_t sum = 0;

    if (length < base_length) {
        // printf("begin: %zu, end: %zu\n", begin, end);
        for(size_t i = begin + 1; i < end; ++ i) {
            array[i] += array[i-1];
        }
    } else {
        #pragma omp task shared(sum)
        {
            sum = prefix_sum_omp_task(array, begin, mid, base_length);
        }
        #pragma omp task
        {
            prefix_sum_omp_task(array, mid, end, base_length);
        }
        #pragma omp taskwait

        #pragma omp parallel for
        for(size_t i = mid; i < end; ++ i) {
            array[i] += sum;
        }
    }
    return array[end - 1];
}

}

void scan_seq(size_t * array, size_t const length) {
    for(size_t i = 1; i < length; ++ i) {
        array[i] += array[i-1];
    }
}


void scan_omp(size_t * array, size_t const length) {
    int nthreads;
    #pragma omp parallel shared(nthreads)
    {
        nthreads = omp_get_num_threads();
    }
    size_t base_length = (length + nthreads - 1) / nthreads;

    prefix_sum_omp_task(array, 0, length, base_length);
}

}
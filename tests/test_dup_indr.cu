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

#include <stdio.h>
#include <ParTI.h>
#include "../src/cudawrap.h"

const short data_0[6] = { 3, 1, 4, 1, 5, 9 };
const short data_1[5] = { 2, 6, 5, 3, 5 };
const short data_2[4] = { 8, 9, 7, 9 };
const short data_3[3] = { 3, 2, 3 };
const short data_4[2] = { 8, 4 };
const short data_5[1] = { 6 };
const short data_6[1] = { -0x3334 };

const short *const header[7] = { data_0, data_1, data_2, data_3, data_4, data_5, data_6 };
const size_t length[7] = { 6, 5, 4, 3, 2, 1, 0 };

/* You may as well write this as a C++11 closure */
static inline size_t get_length(size_t idx) {
    return length[idx];
}

static void print_data(const char *title, const short *const *data) {
    printf("%s at %p:\n", title, data);
    for(size_t i = 0; i < 7; ++i) {
        printf("data_%zu at %p:", i, data[i]);
        for(size_t j = 0; j < length[i]; ++j) {
            printf(" %d", (int) data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_dev_data(const char *title, const short *const *dev_data) {
    printf("%s at %p:\n", title, dev_data);
    const short **data = new const short *[7];
    cudaMemcpy(data, dev_data, 7 * sizeof *dev_data, cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < 7; ++i) {
        printf("data_%zu at %p: <pointer to GPU memory>", i, data[i]);
        printf("\n");
    }
    delete[] data;
    printf("\n");
}

static void compare_data(const short *const *data1, const short *const *data2) {
    for(size_t i = 0; i < 7; ++i) {
        for(size_t j = 0; j < length[i]; ++j) {
            if(data1[i][j] != data2[i][j]) {
                abort();
            }
        }
    }
}

int main() {
    print_data("Original data", header);

    short **dev_data;
    sptAssert(sptCudaDuplicateMemoryIndirect(&dev_data, header, 7, length, cudaMemcpyHostToDevice) == 0);

    print_dev_data("Device data", dev_data);

    short **copyback_data;
    sptAssert(sptCudaDuplicateMemoryIndirect(&copyback_data, dev_data, 7, length, cudaMemcpyDeviceToHost) == 0);

    cudaFree(dev_data);

    print_data("Copyback data", copyback_data);

    free(copyback_data);

    return 0;
}

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

int main(int argc, char *argv[]) {
    int i;
    if(argc == 1) {
        cudaDeviceReset();
        printf("CUDA device reset.\n");
    } else {
        for(i = 1; i < argc; ++i) {
            int dev_id;
            if(sscanf(argv[i], "%d", &dev_id) == 1) {
                cudaSetDevice(i);
                cudaDeviceReset();
                printf("CUDA device #%d reset.\n", dev_id);
            } else {
                fprintf(stderr, "Invalid CUDA device ID: %s\n", argv[i]);
            }
        }
    }
    return 0;
}

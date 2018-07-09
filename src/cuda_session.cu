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

#include <ParTI/session.hpp>
#include <cstdio>
#include <ParTI/device.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memnode.hpp>

namespace pti {

void Session::detect_cuda_devices() {
    cudaError_t error;
    int num_cuda_devices = 0;
    error = cudaGetDeviceCount(&num_cuda_devices);
    if(error) {
        std::fprintf(stderr, "Warning: can not detect CUDA devices\n");
    }

    for(int i = 0; i < num_cuda_devices; ++i) {
        CudaMemNode* cuda_mem_node = new CudaMemNode(i);
        int cuda_mem_node_id = add_mem_node(cuda_mem_node);
        CudaDevice* cuda_device = new CudaDevice(i, cuda_mem_node_id);
        add_device(cuda_device);
    }
}

}

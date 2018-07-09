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

#include <ParTI/device.hpp>
#include <ParTI/error.hpp>

namespace pti {

CudaDevice::CudaDevice(int cuda_device, int mem_node) {
    cudaError_t error;

    struct cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, cuda_device);
    ptiCheckCUDAError(error);

    this->name = "CUDA: ";
    this->name += prop.name;
    this->mem_node = mem_node;
    this->cuda_device = cuda_device;
}

}

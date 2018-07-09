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

#ifndef PTI_DEVICE_INCLUDED
#define PTI_DEVICE_INCLUDED

#include <string>

namespace pti {

struct MemNode;

/// A computing device
struct Device {

    std::string name;
    int device_id;
    int mem_node;

    virtual ~Device() {
    }

};

struct CpuDevice : public Device {

    CpuDevice(int cpu_core, int mem_node);

    int cpu_core;

};

struct CudaDevice : public Device {

    CudaDevice(int cuda_device, int mem_node);

    void* GetCusolverDnHandle();
    void* GetCusolverSpHandle();
    void* GetCublasHandle();

    int cuda_device;

};

// Not used
struct ClDevice : public Device {

    ClDevice(void* cl_device, int mem_node);

    void* cl_device;

};

}

#endif

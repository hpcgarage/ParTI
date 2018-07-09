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
#include <string>

namespace pti {

CpuDevice::CpuDevice(int cpu_core, int mem_node) {
    this->name = "CPU: Core ";
    this->name += std::to_string(cpu_core);
    this->mem_node = mem_node;
    this->cpu_core = cpu_core;
}

}

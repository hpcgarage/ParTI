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

#ifndef PTI_SESSION_INCLUDED
#define PTI_SESSION_INCLUDED

#include <cstddef>
#include <memory>
#include <vector>

namespace pti {

struct Device;

struct MemNode;

struct Session {

    Session();
    ~Session();
    void print_devices() const;

    std::vector<Device*> devices;
    std::vector<MemNode*> mem_nodes;

private:
    void detect_devices();
    void detect_cuda_devices();
    void detect_cl_devices();
    int add_device(Device* device);
    int add_mem_node(MemNode* mem_node);

};

extern Session session;

constexpr int cpu = 0;

}

#endif

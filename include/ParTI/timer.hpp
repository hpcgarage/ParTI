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

#ifndef PTI_TIMER_INCLUDED
#define PTI_TIMER_INCLUDED

#include <cstddef>
#include <ParTI/memblock.hpp>

struct timespec;

namespace pti {

struct CudaDevice;

struct Timer {

    Timer();
    Timer(int device);
    ~Timer();
    void start();
    void stop();
    double elapsed_time() const;
    double print_elapsed_time(char const* name) const;

private:

    int device;
    CudaDevice* cuda_dev;
#ifndef _WIN32
    struct timespec start_timespec;
    struct timespec stop_timespec;
#else
    int64_t start_perfcount;
    int64_t stop_perfcount;
#endif
    void* cuda_start_event;
    void* cuda_stop_event;

    void cuda_init();
    void cuda_fini();
    void cuda_start();
    void cuda_stop();
    double cuda_elapsed_time() const;

};

void tick();

double tock();

double tock(char const* name);

}

#endif

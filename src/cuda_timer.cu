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

#include <ParTI/timer.hpp>
#include <cstdio>
#include <ParTI/device.hpp>
#include <ParTI/error.hpp>
#include <ParTI/session.hpp>

namespace pti {

void Timer::cuda_init() {
    cudaError_t error;

    error = cudaEventCreate((cudaEvent_t*) &cuda_start_event);
    ptiCheckCUDAError(error);

    error = cudaEventCreate((cudaEvent_t*) &cuda_stop_event);
    ptiCheckCUDAError(error);
}

void Timer::cuda_fini() {
    cudaEventDestroy((cudaEvent_t) cuda_start_event);
    cudaEventDestroy((cudaEvent_t) cuda_stop_event);
}

void Timer::cuda_start() {
    cudaError_t error;

    int old_device;
    error = cudaGetDevice(&old_device);
    ptiCheckCUDAError(error);

    error = cudaSetDevice(cuda_dev->cuda_device);
    ptiCheckCUDAError(error);

    error = cudaEventRecord((cudaEvent_t) cuda_start_event);
    ptiCheckCUDAError(error);

    error = cudaEventSynchronize((cudaEvent_t) cuda_start_event);
    ptiCheckCUDAError(error);

    error = cudaSetDevice(old_device);
    ptiCheckCUDAError(error);
}

void Timer::cuda_stop() {
    cudaError_t error;

    int old_device;
    error = cudaGetDevice(&old_device);
    ptiCheckCUDAError(error);

    error = cudaSetDevice(cuda_dev->cuda_device);
    ptiCheckCUDAError(error);

    error = cudaEventRecord((cudaEvent_t) cuda_stop_event);
    ptiCheckCUDAError(error);

    error = cudaEventSynchronize((cudaEvent_t) cuda_stop_event);
    ptiCheckCUDAError(error);

    error = cudaSetDevice(old_device);
    ptiCheckCUDAError(error);
}

double Timer::cuda_elapsed_time() const {
    float elapsed;
    if(cudaEventElapsedTime(&elapsed, (cudaEvent_t) cuda_start_event, (cudaEvent_t) cuda_stop_event) != 0) {
        return NAN;
    }
    return elapsed * 1e-3;
}

}

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
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/session.hpp>

#ifndef _WIN32
#include <time.h>
#else
#include <windows.h>
#endif

namespace pti {

Timer::Timer() {
    this->device = cpu;
    cuda_dev = nullptr;
}

Timer::Timer(int device) {
    this->device = device;
    cuda_dev = dynamic_cast<CudaDevice*>(session.devices[device]);
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_init();
#else
        ptiCheckCUDAError(true);
#endif
    }
}

Timer::~Timer() {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_fini();
#endif
    }
}

void Timer::start() {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_start();
#else
        ptiCheckCUDAError(true);
#endif
    } else {
#ifndef _WIN32
        clock_gettime(CLOCK_MONOTONIC, &start_timespec);
#else
        LARGE_INTEGER performance_counter;
        BOOL result = QueryPerformanceCounter(&performance_counter);
        ptiCheckError(!result, ERR_UNKNOWN, "No high resolution timer available on this system");
        start_perfcount = performance_counter.QuadPart;
#endif
    }
}

void Timer::stop() {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_stop();
#else
        ptiCheckCUDAError(true);
#endif
    } else {
#ifndef _WIN32
        clock_gettime(CLOCK_MONOTONIC, &stop_timespec);
#else
        LARGE_INTEGER performance_counter;
        BOOL result = QueryPerformanceCounter(&performance_counter);
        ptiCheckError(!result, ERR_UNKNOWN, "No high resolution timer available on this system");
        stop_perfcount = performance_counter.QuadPart;
#endif
    }
}

double Timer::elapsed_time() const {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        return cuda_elapsed_time();
#else
        ptiCheckCUDAError(true);
#endif
    } else {
#ifndef _WIN32
        return stop_timespec.tv_sec - start_timespec.tv_sec
            + (stop_timespec.tv_nsec - start_timespec.tv_nsec) * 1e-9;
#else
        LARGE_INTEGER performance_frequency;
        BOOL result = QueryPerformanceFrequency(&performance_frequency);
        ptiCheckError(!result || performance_frequency.QuadPart == 0, ERR_UNKNOWN, "No high resolution timer available on this system");
        return double(stop_perfcount - start_perfcount) / performance_frequency.QuadPart;
#endif
    }
}

double Timer::print_elapsed_time(char const* name) const {
    double elapsed_time = this->elapsed_time();
    std::fprintf(stderr, "[%s]: %.9lf s spent on device \"%s\"\n", name, elapsed_time, session.devices[this->device]->name.c_str());
    std::fflush(stderr);
    return elapsed_time;
}

static Timer default_timer;

void tick() {
    default_timer.start();
}

double tock() {
    default_timer.stop();
    return default_timer.elapsed_time();
}

double tock(char const* name) {
    default_timer.stop();
    return default_timer.print_elapsed_time(name);
}

}

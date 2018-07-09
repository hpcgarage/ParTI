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

#include <ParTI/error.hpp>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace pti {

void Error::print_error() const {
    if(err_file) {
        std::fprintf(stderr, "Error 0x%08x at %s:%u, %s\n",
            err_code, err_file, err_line, what());
    } else {
        std::fprintf(stderr, "Error 0x%08x, %s\n",
            err_code, what());
    }
}

OSError::OSError() :
    Error(errno, std::strerror(errno)) {
}

OSError::OSError(int code) :
    Error(code, std::strerror(code)) {
}

OSError::OSError(int code, char const* file, unsigned line) :
    Error(code, std::strerror(code), file, line) {
}

OSError::OSError(char const* file, unsigned line) :
    Error(errno, std::strerror(errno), file, line) {
}

#ifndef PARTI_USE_CUDA

CUDAError::CUDAError() :
    Error(3, "CUDA support not enabled at compile time") {
}

CUDAError::CUDAError(int code) :
    Error(code, "CUDA support not enabled at compile time") {
}

CUDAError::CUDAError(int code, char const* file, unsigned line) :
    Error(code, "CUDA support not enabled at compile time", file, line) {
}

CUDAError::CUDAError(char const* file, unsigned line) :
    Error(3, "CUDA support not enabled at compile time", file, line) {
}

#endif

}

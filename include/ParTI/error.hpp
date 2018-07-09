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

#ifndef PTI_ERROR_INCLUDED
#define PTI_ERROR_INCLUDED

#include <stdexcept>
#include <string>

#ifndef unlikely
#ifndef _MSC_VER
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define unlikely(x) (x)
#endif
#endif

namespace pti {

class Error : public std::runtime_error {
    int err_code;
    char const* err_file;
    unsigned err_line;
public:
    explicit Error(int code, char const* msg) :
        std::runtime_error(msg) {
        err_code = code;
        err_file = nullptr;
        err_line = 0;
    }
    explicit Error(int code, char const* msg, char const* file, unsigned line) :
        std::runtime_error(msg) {
        err_code = code;
        err_file = file;
        err_line = line;
    }
    int code() const {
        return err_code;
    }
    void print_error() const;
};

class OSError : public Error {
public:
    explicit OSError();
    explicit OSError(int code);
    explicit OSError(int code, char const* file, unsigned line);
    explicit OSError(char const* file, unsigned line);
};

class CUDAError : public Error {
public:
    explicit CUDAError();
    explicit CUDAError(int code);
    explicit CUDAError(int code, char const* file, unsigned line);
    explicit CUDAError(char const* file, unsigned line);
};

}

#define ptiCheckError(cond, code, msg) \
    if(unlikely((cond))) { \
        throw ::pti::Error(code, msg, __FILE__, __LINE__); \
    }

#define ptiCheckOSError(cond) \
    if(unlikely((cond))) { \
        throw ::pti::OSError(__FILE__, __LINE__); \
    }

#define ptiCheckCUDAError(cond) \
    if(unlikely((cond))) { \
        throw ::pti::CUDAError(__FILE__, __LINE__); \
    }

#endif

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

namespace pti {

CUDAError::CUDAError() :
    Error((int) cudaGetLastError(), cudaGetErrorString(cudaGetLastError())) {
}

CUDAError::CUDAError(int code) :
    Error(code, cudaGetErrorString((cudaError_t) code)) {
}

CUDAError::CUDAError(int code, char const* file, unsigned line) :
    Error(code, cudaGetErrorString((cudaError_t) code), file, line) {
}

CUDAError::CUDAError(char const* file, unsigned line) :
    Error((int) cudaGetLastError(), cudaGetErrorString(cudaGetLastError()), file, line) {
}

}

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

#ifndef PARTI_CUDAWRAP_H
#define PARTI_CUDAWRAP_H

#ifdef PARTI_USE_CUDA

#include <cuda_runtime.h>
#include "error/error.h"

extern "C" {
int spt_CudaDuplicateMemoryGenerics(void **dest, const void *src, size_t size, int direction);
}

template <class T>
static inline int sptCudaDuplicateMemory(T **dest, const T *src, size_t size, int direction) {
    return spt_CudaDuplicateMemoryGenerics((void **) dest, src, size, direction);
}

template <class T>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T **src, size_t nmemb, size_t size, int direction) {
    int result;
    T **host_src;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        host_src = const_cast<T **>(src);
        break;
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDeviceToDevice:
        result = sptCudaDuplicateMemory(&host_src, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckError(result, "sptCudaDuplicateMemoryIndirect", NULL);
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "spt_CudaDuplicateMemory", "Unknown memory copy kind");
    }

    T **host_dest = new T *[nmemb];
    for(size_t i = 0; i < nmemb; ++i) {
        result = sptCudaDuplicateMemory(&host_dest[i], host_src[i], size, direction);
        spt_CheckError(result, "sptCudaDuplicateMemoryIndirect", NULL);
    }

    switch(direction) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToDevice:
        result = sptCudaDuplicateMemory(dest, host_dest, nmemb * sizeof (void *), cudaMemcpyHostToDevice);
        spt_CheckError(result, "sptCudaDuplicateMemoryIndirect", NULL);
        delete[] host_dest;
        break;
    case cudaMemcpyDeviceToHost:
        *dest = host_dest;
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "spt_CudaDuplicateMemory", "Unknown memory copy kind");
    }

    switch(direction) {
    case cudaMemcpyHostToDevice:
        break;
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDeviceToDevice:
        free(host_src);
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "spt_CudaDuplicateMemory", "Unknown memory copy kind")
    }

    return 0;
}

template <class T, class F>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T **src, size_t nmemb, F size, int direction) {
    int result;
    T **host_src;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        host_src = const_cast<T **>(src);
        break;
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDeviceToDevice:
        result = sptCudaDuplicateMemory(&host_src, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckError(result, "sptCudaDuplicateMemoryIndirect", NULL);
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "spt_CudaDuplicateMemory", "Unknown memory copy kind");
    }

    T **host_dest = new T *[nmemb];
    for(size_t i = 0; i < nmemb; ++i) {
        result = sptCudaDuplicateMemory(&host_dest[i], host_src[i], size(i), direction);
        spt_CheckError(result, "sptCudaDuplicateMemoryIndirect", NULL);
    }

    switch(direction) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToDevice:
        result = sptCudaDuplicateMemory(dest, host_dest, nmemb * sizeof (void *), cudaMemcpyHostToDevice);
        spt_CheckError(result, "sptCudaDuplicateMemoryIndirect", NULL);
        delete[] host_dest;
        break;
    case cudaMemcpyDeviceToHost:
        *dest = host_dest;
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "spt_CudaDuplicateMemory", "Unknown memory copy kind");
    }

    switch(direction) {
    case cudaMemcpyHostToDevice:
        break;
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDeviceToDevice:
        free(host_src);
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "spt_CudaDuplicateMemory", "Unknown memory copy kind");
    }

    return 0;
}

#endif

#endif

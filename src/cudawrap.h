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

#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "error/error.h"

extern "C" {
int spt_CudaDuplicateMemoryGenerics(void **dest, const void *src, size_t size, int direction);
}

template <class T>
static inline int sptCudaDuplicateMemory(T **dest, const T *src, size_t size, int direction) {
    return spt_CudaDuplicateMemoryGenerics((void **) dest, src, size, direction);
}

/* `length` as a constant */
template <class T>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, size_t nmemb, size_t length, int direction) {
    int result;

    size_t total_size = nmemb * sizeof (T *) + nmemb * length * sizeof (T);

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc((void **) &head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) (head + nmemb);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, src[i], length * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body += length;
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) (head + nmemb);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, tmp_head[i], length * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body += length;
        }
        assert((char *) head + total_size == (char *) body);

        delete[] tmp_head;

        break;

    default:
        spt_CheckError(SPTERR_UNKNOWN, "sptCudaDuplicateMemoryIndirect", "Unknown memory copy kind");
    }

    *dest = head;

    return 0;
}

/* `length` as an array[nmemb] of size_t */
template <class T>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, size_t nmemb, const size_t length[], int direction) {
    int result;

    size_t total_size = nmemb * sizeof (T *);
    for(size_t i = 0; i < nmemb; ++i) {
        total_size += length[i] * sizeof (T);
    }

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc(&head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) (head + nmemb);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, src[i], length[i] * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body += length[i];
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) (head + nmemb);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, tmp_head[i], length[i] * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body += length[i];
        }
        assert((char *) head + total_size == (char *) body);

        delete[] tmp_head;

        break;

    default:
        spt_CheckError(SPTERR_UNKNOWN, "sptCudaDuplicateMemoryIndirect", "Unknown memory copy kind");
    }

    *dest = head;

    return 0;
}

/* `length` as a closure: (size_t) -> size_t */
template <class T, class Fn>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, size_t nmemb, Fn length, int direction) {
    int result;

    size_t total_size = nmemb * sizeof (T *);
    for(size_t i = 0; i < nmemb; ++i) {
        total_size += length(i) * sizeof (T);
    }

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc(&head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) (head + nmemb);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            size_t this_size = length(i);
            result = cudaMemcpy(body, src[i], this_size * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body += this_size;
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) (head + nmemb);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            size_t this_size = length(i);
            result = cudaMemcpy(body, tmp_head[i], this_size * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body += this_size;
        }
        assert((char *) head + total_size == (char *) body);

        delete[] tmp_head;

        break;

    default:
        spt_CheckError(SPTERR_UNKNOWN, "sptCudaDuplicateMemoryIndirect", "Unknown memory copy kind");
    }

    *dest = head;

    return 0;
}

#endif

#endif

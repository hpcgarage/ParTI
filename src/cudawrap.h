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

typedef struct cusparseContext *cusparseHandle_t;
typedef struct cusolverSpContext *cusolverSpHandle_t;

extern "C" {

int spt_cusparseCreate(cusparseHandle_t *handle);
int spt_cusolverSpCreate(cusolverSpHandle_t *handle);

int spt_CudaDuplicateMemoryGenerics(void **dest, const void *src, size_t size, int direction);
int spt_CudaDuplicateMemoryGenericsAsync(void **dest, const void *src, size_t size, int direction, cudaStream_t stream);

}

template <class T>
static inline int sptCudaDuplicateMemory(T **dest, const T *src, size_t size, int direction) {
    return spt_CudaDuplicateMemoryGenerics((void **) dest, src, size, direction);
}

template <class T>
static inline int sptCudaDuplicateMemoryAsync(T **dest, const T *src, size_t size, int direction) {
    return spt_CudaDuplicateMemoryGenericsAsync((void **) dest, src, size, direction);
}

static size_t spt_cudaGetAlignedSize(size_t size, bool on_gpu) {
    if(size != 0) {
        if(on_gpu) {
            return ((size - 1) / 256 + 1) * 256;
        } else {
            return ((size - 1) / 16 + 1) * 16;
        }
    } else {
        return 0;
    }
}

/* `length` as a constant */
template <class T>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, size_t nmemb, size_t length, int direction) {
    int result;

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    size_t head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    size_t total_size = head_size + nmemb * spt_cudaGetAlignedSize(length * sizeof (T), gpu_align);

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc((void **) &head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, src[i], length * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, tmp_head[i], length * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length * sizeof (T), gpu_align));
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

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    size_t head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    size_t total_size = head_size;
    for(size_t i = 0; i < nmemb; ++i) {
        total_size += spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align);
    }

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc(&head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, src[i], length[i] * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, tmp_head[i], length[i] * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align));
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



/* `length` as a constant */
template <class T>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, sptIndex nmemb, sptNnzIndex length, int direction) {
    int result;

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    sptNnzIndex head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    sptNnzIndex total_size = head_size + nmemb * spt_cudaGetAlignedSize(length * sizeof (T), gpu_align);

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc((void **) &head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(sptIndex i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, src[i], length * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(sptIndex i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, tmp_head[i], length * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length * sizeof (T), gpu_align));
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
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, sptIndex nmemb, const sptIndex length[], int direction) {
    int result;

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    sptNnzIndex head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    sptNnzIndex total_size = head_size;
    for(sptIndex i = 0; i < nmemb; ++i) {
        total_size += spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align);
    }

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc(&head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(sptIndex i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, src[i], length[i] * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(sptIndex i = 0; i < nmemb; ++i) {
            result = cudaMemcpy(body, tmp_head[i], length[i] * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align));
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



/* Async memcpys */
/* `length` as a constant */
template <class T>
inline int sptCudaDuplicateMemoryIndirectAsync(T ***dest, const T *const *src, size_t nmemb, size_t length, int direction, cudaStream_t stream) {
    int result;

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    size_t head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    size_t total_size = head_size + nmemb * spt_cudaGetAlignedSize(length * sizeof (T), gpu_align);

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc((void **) &head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpyAsync(body, src[i], length * sizeof (T), cudaMemcpyHostToDevice, stream);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpyAsync(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice, stream);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpyAsync(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost, stream);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpyAsync(body, tmp_head[i], length * sizeof (T), cudaMemcpyDeviceToHost, stream);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length * sizeof (T), gpu_align));
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
inline int sptCudaDuplicateMemoryIndirectAsync(T ***dest, const T *const *src, size_t nmemb, const size_t length[], int direction, cudaStream_t stream) {
    int result;

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    size_t head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    size_t total_size = head_size;
    for(size_t i = 0; i < nmemb; ++i) {
        total_size += spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align);
    }

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc(&head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpyAsync(body, src[i], length[i] * sizeof (T), cudaMemcpyHostToDevice, stream);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpyAsync(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice, stream);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpyAsync(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost, stream);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            result = cudaMemcpyAsync(body, tmp_head[i], length[i] * sizeof (T), cudaMemcpyDeviceToHost, stream);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(length[i] * sizeof (T), gpu_align));
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



#if 0
/* `length` as a closure: (size_t) -> size_t */
template <class T, class Fn>
inline int sptCudaDuplicateMemoryIndirect(T ***dest, const T *const *src, size_t nmemb, Fn length, int direction) {
    int result;

    bool gpu_align = direction == cudaMemcpyHostToDevice;
    size_t head_size = spt_cudaGetAlignedSize(nmemb * sizeof (T *), gpu_align);
    size_t total_size = head_size;
    for(size_t i = 0; i < nmemb; ++i) {
        total_size += spt_cudaGetAlignedSize(length(i) * sizeof (T), gpu_align);
    }

    T **head;
    T *body;
    T **tmp_head;

    switch(direction) {
    case cudaMemcpyHostToDevice:
        result = cudaMalloc(&head, total_size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];

        for(size_t i = 0; i < nmemb; ++i) {
            size_t this_size = length(i);
            result = cudaMemcpy(body, src[i], this_size * sizeof (T), cudaMemcpyHostToDevice);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            tmp_head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(this_size * sizeof (T), gpu_align));
        }
        assert((char *) head + total_size == (char *) body);

        result = cudaMemcpy(head, tmp_head, nmemb * sizeof (T *), cudaMemcpyHostToDevice);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        delete[] tmp_head;

        break;

    case cudaMemcpyDeviceToHost:
        head = (T **) malloc(total_size);
        spt_CheckOSError(head == NULL, "sptCudaDuplicateMemoryIndirect");
        body = (T *) ((char *) head + head_size);

        tmp_head = new T*[nmemb];
        result = cudaMemcpy(tmp_head, src, nmemb * sizeof (T *), cudaMemcpyDeviceToHost);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

        for(size_t i = 0; i < nmemb; ++i) {
            size_t this_size = length(i);
            result = cudaMemcpy(body, tmp_head[i], this_size * sizeof (T), cudaMemcpyDeviceToHost);
            spt_CheckCudaError(result != 0, "sptCudaDuplicateMemoryIndirect");

            head[i] = body;
            body = (T *) ((char *) body + spt_cudaGetAlignedSize(this_size * sizeof (T), gpu_align));
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

#endif

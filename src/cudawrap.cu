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

#include <ParTI.h>
#include <stdlib.h>
#include "error/error.h"
#include "cudawrap.h"
#include <cusparse.h>
#include <cusolverSp.h>

int sptCudaSetDevice(int device) {
    return (int) cudaSetDevice(device);
}

int sptCudaGetLastError(void) {
    return (int) cudaGetLastError();
}


int spt_CudaDuplicateMemoryGenerics(void **dest, const void *src, size_t size, int direction) {
    int result;
    switch(direction) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToDevice:
        result = cudaMalloc(dest, size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemory");
        break;
    case cudaMemcpyDeviceToHost:
        *dest = malloc(size);
        spt_CheckOSError(*dest == NULL, "sptCudaDuplicateMemory");
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "sptCudaDuplicateMemory", "Unknown memory copy kind")
    }
    result = cudaMemcpy(*dest, src, size, (cudaMemcpyKind) direction);
    spt_CheckCudaError(result != 0, "sptCudaDuplicateMemory");
    return 0;
}


int spt_CudaDuplicateMemoryGenericsAsync(void **dest, const void *src, size_t size, int direction, cudaStream_t stream) {
    int result;
    switch(direction) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToDevice:
        result = cudaMalloc(dest, size);
        spt_CheckCudaError(result != 0, "sptCudaDuplicateMemory");
        break;
    case cudaMemcpyDeviceToHost:
        *dest = malloc(size);
        spt_CheckOSError(*dest == NULL, "sptCudaDuplicateMemory");
        break;
    default:
        spt_CheckError(SPTERR_UNKNOWN, "sptCudaDuplicateMemory", "Unknown memory copy kind")
    }
    result = cudaMemcpyAsync(*dest, src, size, (cudaMemcpyKind) direction, stream);
    spt_CheckCudaError(result != 0, "sptCudaDuplicateMemory");
    return 0;
}

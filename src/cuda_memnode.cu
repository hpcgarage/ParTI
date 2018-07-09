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

#include <ParTI/memnode.hpp>
#include <cstdlib>
#include <ParTI/error.hpp>

namespace pti {

CudaMemNode::CudaMemNode(int cuda_device) {
    this->cuda_device = cuda_device;
    char const* env_pti_memcpy_profile = std::getenv("PTI_MEMCPY_PROFILE");
    if(env_pti_memcpy_profile != nullptr && std::strcmp(env_pti_memcpy_profile, "1") == 0) {
        memcpy_profiling = true;
    }
}

void* CudaMemNode::malloc(size_t size) {
    cudaError_t error;

    int old_device;
    error = cudaGetDevice(&old_device);
    ptiCheckCUDAError(error);

    error = cudaSetDevice(cuda_device);
    ptiCheckCUDAError(error);

    void* ptr;
    error = cudaMalloc(&ptr, size);
    ptiCheckCUDAError(error);

    if(malloc_profiling) {
        profile(ptr, size);
        std::fprintf(stderr, "[CudaMemNode]: malloc(%zu) = %p,\t%s used, %s max\n", size, ptr, bytes_allocated_str().c_str(), max_bytes_allocated_str().c_str());
    }

    error = cudaSetDevice(old_device);
    ptiCheckCUDAError(error);

    return ptr;
}

void* CudaMemNode::realloc(void*, size_t) {
    throw std::bad_alloc();
}

void CudaMemNode::free(void* ptr) {
    cudaError_t error;

    int old_device;
    error = cudaGetDevice(&old_device);
    ptiCheckCUDAError(error);

    error = cudaSetDevice(cuda_device);
    ptiCheckCUDAError(error);

    error = cudaFree(ptr);
    ptiCheckCUDAError(error);

    if(malloc_profiling) {
        size_t oldsize = profile(ptr, 0);
        std::fprintf(stderr, "[CudaMemNode]: free(%p[%zu]),\t%s used, %s max\n", ptr, oldsize, bytes_allocated_str().c_str(), max_bytes_allocated_str().c_str());
    }

    error = cudaSetDevice(old_device);
    ptiCheckCUDAError(error);
}

void CudaMemNode::memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size) {
    cudaEvent_t cuda_start_event, cuda_stop_event;
    cudaError_t error;
    if(memcpy_profiling) {
        error = cudaEventCreate(&cuda_start_event);
        ptiCheckCUDAError(error);
        error = cudaEventCreate(&cuda_stop_event);
        ptiCheckCUDAError(error);
    }
    if(CpuMemNode* cpu_dest_node = dynamic_cast<CpuMemNode*>(&dest_node)) {
        error = cudaSetDevice(cuda_device);
        ptiCheckCUDAError(error);
        if(memcpy_profiling) {
            error = cudaEventRecord(cuda_start_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_start_event);
            ptiCheckCUDAError(error);
        }
        error = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
        ptiCheckCUDAError(error);
        if(memcpy_profiling) {
            error = cudaEventRecord(cuda_stop_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_stop_event);
            ptiCheckCUDAError(error);
            float elapsed;
            error = cudaEventElapsedTime(&elapsed, cuda_start_event, cuda_stop_event);
            ptiCheckCUDAError(error);
            cudaEventDestroy(cuda_start_event);
            cudaEventDestroy(cuda_stop_event);
            std::fprintf(stderr, "[CudaMemNode DtoH]: %.9lf s spent on devices\n", elapsed * 1e-3);
        }
    } else if(CudaMemNode* cuda_dest_node = dynamic_cast<CudaMemNode*>(&dest_node)) {
        if(memcpy_profiling) {
            error = cudaSetDevice(cuda_device);
            ptiCheckCUDAError(error);
            error = cudaEventRecord(cuda_start_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_start_event);
            ptiCheckCUDAError(error);
        }
        error = cudaMemcpyPeer(dest, cuda_dest_node->cuda_device, src, cuda_device, size);
        ptiCheckCUDAError(error);
        if(memcpy_profiling) {
            error = cudaEventRecord(cuda_stop_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_stop_event);
            ptiCheckCUDAError(error);
            float elapsed;
            error = cudaEventElapsedTime(&elapsed, cuda_start_event, cuda_stop_event);
            ptiCheckCUDAError(error);
            cudaEventDestroy(cuda_start_event);
            cudaEventDestroy(cuda_stop_event);
            std::fprintf(stderr, "[CudaMemNode DtoD]: %.9lf s spent on devices\n", elapsed * 1e-3);
        }
    } else {
        ptiCheckError(true, 1, "Unknown memory node type");
    }
}

void CudaMemNode::memcpy_from(void* dest, void* src, MemNode& src_node, size_t size) {
    cudaEvent_t cuda_start_event, cuda_stop_event;
    cudaError_t error;
    if(memcpy_profiling) {
        error = cudaEventCreate(&cuda_start_event);
        ptiCheckCUDAError(error);
        error = cudaEventCreate(&cuda_stop_event);
        ptiCheckCUDAError(error);
    }
    if(CpuMemNode* cpu_src_node = dynamic_cast<CpuMemNode*>(&src_node)) {
        error = cudaSetDevice(cuda_device);
        ptiCheckCUDAError(error);
        if(memcpy_profiling) {
            error = cudaEventRecord(cuda_start_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_start_event);
            ptiCheckCUDAError(error);
        }
        error = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
        ptiCheckCUDAError(error);
        if(memcpy_profiling) {
            error = cudaEventRecord(cuda_stop_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_stop_event);
            ptiCheckCUDAError(error);
            float elapsed;
            error = cudaEventElapsedTime(&elapsed, cuda_start_event, cuda_stop_event);
            ptiCheckCUDAError(error);
            cudaEventDestroy(cuda_start_event);
            cudaEventDestroy(cuda_stop_event);
            std::fprintf(stderr, "[CudaMemNode HtoD]: %.9lf s spent on devices\n", elapsed * 1e-3);
        }
    } else if(CudaMemNode* cuda_src_node = dynamic_cast<CudaMemNode*>(&src_node)) {
        if(memcpy_profiling) {
            error = cudaSetDevice(cuda_device);
            ptiCheckCUDAError(error);
            error = cudaEventRecord(cuda_start_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_start_event);
            ptiCheckCUDAError(error);
        }
        error = cudaMemcpyPeer(dest, cuda_device, src, cuda_src_node->cuda_device, size);
        ptiCheckCUDAError(error);
        if(memcpy_profiling) {
            error = cudaEventRecord(cuda_stop_event);
            ptiCheckCUDAError(error);
            error = cudaEventSynchronize(cuda_stop_event);
            ptiCheckCUDAError(error);
            float elapsed;
            error = cudaEventElapsedTime(&elapsed, cuda_start_event, cuda_stop_event);
            ptiCheckCUDAError(error);
            cudaEventDestroy(cuda_start_event);
            cudaEventDestroy(cuda_stop_event);
            std::fprintf(stderr, "[CudaMemNode DtoD]: %.9lf s spent on devices\n", elapsed * 1e-3);
        }
    } else {
        ptiCheckError(true, 1, "Unknown memory node type");
    }
}

}

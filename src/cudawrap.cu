#include <SpTOL.h>

int sptCudaSetDevice(int device) {
    return (int) cudaSetDevice(device);
}

int sptCudaGetLastError(void) {
    return (int) cudaGetLastError();
}

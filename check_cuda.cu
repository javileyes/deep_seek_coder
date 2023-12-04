#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        printf("GPU is available\n");
    } else {
        printf("GPU is not available\n");
    }
    return 0;
}


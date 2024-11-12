#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckError() {                                  \
    cudaError_t e=cudaGetLastError();                        \
    if(e!=cudaSuccess) {                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
}

#endif

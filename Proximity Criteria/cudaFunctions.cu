// cudafunctions.cu

#include <cuda_runtime.h>  // CUDA runtime API, providing functions for managing GPU resources and launching kernels
#include <math.h>          // Math library for distance calculations
#include <float.h>         // Library for defining float limits, e.g., FLT_MAX for representing a large float value

// CUDA kernel that calculates distances between all pairs of points
__global__ void calculateDistances(float *x, float *y, float *distances, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate the global index of the current thread within the grid
    if (i < N) {  // Ensure the thread index is within the bounds of the array size
        for (int j = 0; j < N; j++) {
            if (i != j) {  // Avoid calculating the distance of a point to itself
                float dx = x[i] - x[j];  // Calculate the difference in x-coordinates
                float dy = y[i] - y[j];  // Calculate the difference in y-coordinates
                distances[i * N + j] = sqrtf(dx * dx + dy * dy);  // Compute the Euclidean distance and store it
            } else {
                distances[i * N + j] = FLT_MAX;  // Set the distance to a large value for self-distance to ignore it
            }
        }
    }
}

// Function that handles the memory management and kernel launch for distance calculations on the GPU
void gpuCalculateDistances(float *x, float *y, float *distances, int N) {
    float *d_x, *d_y, *d_distances;  // Device pointers for the x, y coordinates, and the distance array

    // Allocate memory on the GPU for the x coordinates, y coordinates, and distances array
    cudaMalloc((void**)&d_x, N * sizeof(float));  // Allocate N floats for x coordinates
    cudaMalloc((void**)&d_y, N * sizeof(float));  // Allocate N floats for y coordinates
    cudaMalloc((void**)&d_distances, N * N * sizeof(float));  // Allocate N x N floats for distances

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);  // Copy x coordinates to GPU
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);  // Copy y coordinates to GPU

    // Define the number of threads per block and the number of blocks needed
    int blockSize = 256;  // Standard block size for CUDA, 256 threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Calculate the number of blocks required to cover all points

    // Launch the CUDA kernel to calculate distances between all pairs of points
    calculateDistances<<<numBlocks, blockSize>>>(d_x, d_y, d_distances, N);  // Execute the kernel on the GPU

    // Synchronize the device to ensure all threads have completed execution before proceeding
    cudaDeviceSynchronize();

    // Copy the calculated distances back from the device (GPU) to the host (CPU)
    cudaMemcpy(distances, d_distances, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated GPU memory to avoid memory leaks
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_distances);
}

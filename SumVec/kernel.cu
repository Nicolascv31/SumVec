#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* A, const int* B, int* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 10000;  
    size_t size = n * sizeof(int);

    
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

   
    for (int i = 0; i < n; i++) {
        h_A[i] = 10;
        h_B[i] = 3;
    }

   
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, n);

    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    
    std::cout << "Resultado: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

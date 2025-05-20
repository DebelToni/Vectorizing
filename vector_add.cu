#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    const int N = 10;
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = float(i);
        h_B[i] = float(i) * 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<1, N>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "C = ";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

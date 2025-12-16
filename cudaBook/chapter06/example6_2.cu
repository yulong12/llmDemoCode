/**
 * @file example6_2.cu
 * @brief 演示流式分块矩阵的实现，包括数据调度和计算重叠的完整过程
 * @details 该案例展示了如何利用流式分块矩阵乘法的原理，将矩阵乘法任务分解为多个小的矩阵块，并行执行，同时重叠数据传输和计算，提高了计算效率。
 * @author zhangyulong (zhangyulong@ict.ac.cn)
 * @version 1.0
 * @date 2023-12-20
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// 宏函数：通用错误检测
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：矩阵块乘法
__global__ void matrixMultiplyBlock(const float *a, const float *b, float *c, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += a[row * M + i] * b[i * K + col];
        }
        c[row * K + col] = sum;
    }
}
int main() {
    const int N = 512; // 矩阵A的行数
    const int M = 512; // 矩阵A的列数和矩阵B的行数
    const int K = 512; // 矩阵B的列数
    const int blockSize = 16; // 线程块大小
    const int blockCount = 4; // 分块数量
    const size_t bytesA = N * M * sizeof(float);
    const size_t bytesB = M * K * sizeof(float);
    const size_t bytesC = N * K * sizeof(float);
    // 主机内存分配
    std::vector<float> hostA(N * M, 1.0f);
    std::vector<float> hostB(M * K, 2.0f);
    std::vector<float> hostC(N * K, 0.0f);
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc(&deviceA, bytesA));
    CUDA_CHECK(cudaMalloc(&deviceB, bytesB));
    CUDA_CHECK(cudaMalloc(&deviceC, bytesC));
    // 分块传输与计算
    dim3 threads(blockSize, blockSize);
    dim3 grid((K + blockSize - 1) / blockSize, (N / blockCount + blockSize - 1) / blockSize);
    cudaStream_t streams[blockCount];
    for (int i = 0; i < blockCount; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    for (int i = 0; i < blockCount; ++i) {
        int offset = i * (N / blockCount) * M;
        CUDA_CHECK(cudaMemcpyAsync(deviceA + offset, hostA.data() + offset, (N / blockCount) * M * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    }
    CUDA_CHECK(cudaMemcpyAsync(deviceB, hostB.data(), bytesB, cudaMemcpyHostToDevice));
    for (int i = 0; i < blockCount; ++i) {
        int offsetA = i * (N / blockCount) * M;
        int offsetC = i * (N / blockCount) * K;
        matrixMultiplyBlock<<<grid, threads, 0, streams[i]>>>(deviceA + offsetA, deviceB, deviceC + offsetC, N / blockCount, M, K);
    }
    for (int i = 0; i < blockCount; ++i) {
        int offset = i * (N / blockCount) * K;
        CUDA_CHECK(cudaMemcpyAsync(hostC.data() + offset, deviceC + offset, (N / blockCount) * K * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }
    for (int i = 0; i < blockCount; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << hostC[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    return 0;
}

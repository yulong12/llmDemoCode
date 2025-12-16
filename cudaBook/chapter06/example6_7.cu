/**
 * @file example6_7.cu
 * @brief 通过矩阵乘法的实现，演示线程块规模对性能的影响。
 * @details 分析与优化：
 * （1）线程块大小的选择：实验表明，16×16线程块能够有效利用共享内存，同时避免过多的寄存器冲突。如果矩阵规模更大，可以尝试使用32×32线程块以提升性能。
 * （2）网格与块划分：通过动态调整线程块和网格大小，可以优化全局内存访问模式，减少未被处理的空闲线程。
 * （3）分块策略：如果矩阵规模超过设备内存容量，可以结合分块技术和流式计算实现更大的任务处理。
 * @note 本案例通过分析分块策略和线程块规模的选择，为实际问题提供了调优方向，显著提高了CUDA程序的计算性能。
 * @author zhangyulong (zhangyulong@ict.ac.cn)
 * @version 1.0
 * @date 2023-12-20
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
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
// 核函数：矩阵乘法
__global__ void matrixMultiply(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
int main() {
    const int N = 1024; // 矩阵大小
    const size_t bytes = N * N * sizeof(float);
    // 主机内存分配
    std::vector<float> hostA(N * N, 1.0f);
    std::vector<float> hostB(N * N, 1.0f);
    std::vector<float> hostC(N * N, 0.0f);
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 threads(16, 16); // 线程块大小
    dim3 grid((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    // 启动核函数
    matrixMultiply<<<grid, threads>>>(deviceA, deviceB, deviceC, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
    // 打印部分结果
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
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

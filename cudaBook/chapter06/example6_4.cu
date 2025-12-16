/**
 * @file example6_4.cu
 * @brief 以向量归约（Reduction）为例，演示如何利用合并操作优化算术强度不足的算法。
 * @detail 该案例通过以下方法提升了算法强度：1，合并操作：将多个线程的计算结果合并为一个，减少内存访问次数。
 * 2，增加计算量：每个线程处理多个元素，提升每次内存访问的计算收益，从而提高算术强度。
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
// 核函数：使用共享内存实现向量归约
__global__ void vectorReduction(const float *input, float *output, int N) {
    extern __shared__ float sharedData[]; // 动态分配共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 将全局内存的数据加载到共享内存
    sharedData[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    // 在共享内存中进行归约操作
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    // 将每个块的归约结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
int main() {
    const int N = 1 << 20; // 向量大小
    const int blockSize = 256; // 每个块的线程数
    const int gridSize = (N + blockSize - 1) / blockSize; // 网格大小
    const size_t bytesInput = N * sizeof(float);
    const size_t bytesOutput = gridSize * sizeof(float);
    // 主机内存分配
    std::vector<float> hostInput(N, 1.0f); // 初始化向量，所有值为1
    std::vector<float> hostOutput(gridSize, 0.0f);
    // 设备内存分配
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytesInput));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytesOutput));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), bytesInput, cudaMemcpyHostToDevice));
    // 启动核函数
    size_t sharedMemoryBytes = blockSize * sizeof(float); // 每个块的共享内存大小
    vectorReduction<<<gridSize, blockSize, sharedMemoryBytes>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytesOutput, cudaMemcpyDeviceToHost));
    // 对主机上的块结果进行最终归约
    float finalResult = 0.0f;
    for (const auto &val : hostOutput) {
        finalResult += val;
    }
    // 打印结果
    std::cout << "Final Reduction Result: " << finalResult << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    return 0;
}


/**
 * @file example10_5.cu
 * @brief 演示如何使用cuRAND库生成伪随机数。
 * @author zhangyulong (zhangyulong@ict.ac.cn)
 * @version 1.0
 * @date 2023-12-20
 */
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
// 核函数：初始化cuRAND状态并生成随机数
__global__ void generateRandomNumbers(float* output, int size, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    // 初始化cuRAND状态
    curandState state;
    curand_init(seed, idx, 0, &state);
    // 生成随机数
    output[idx] = curand_uniform(&state);
}
int main() {
    const int arraySize = 1024;
    const int blockSize = 256;
    const int numBlocks = (arraySize + blockSize - 1) / blockSize;
    // 分配主机和设备内存
    float* hostArray = new float[arraySize];
    float* deviceArray;
    cudaMalloc(&deviceArray, arraySize * sizeof(float));
    // 调用核函数
    generateRandomNumbers<<<numBlocks, blockSize>>>(deviceArray, arraySize, time(nullptr));
    // 将结果拷回主机
    cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    // 输出部分随机数
    std::cout << "Generated Random Numbers:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    delete[] hostArray;
    cudaFree(deviceArray);
    return 0;
/**
 * @file example6_8.cu
 * @brief 实现一个动态调整并行规模的矩阵加法，演示如何根据数据规模调整线程块和网格的大小。
 * 分析与优化：
 * （1）动态调整线程块大小：代码中通过判断数据规模动态调整线程块的大小，在大规模数据时使用更大的线程块以提高吞吐量。
 * （2）分块策略的灵活性：通过灵活调整网格大小和线程块数量，使得不同规模的数据集都能高效运行。
 * （3）性能提升：动态调整后的并行规模优化了GPU资源的使用效率，确保了程序的高效性和通用性。
 * 本案例通过动态调整并行规模的方法，演示了如何适配不同数据集，提供了高效且灵活的解决方案。这种技术在实际开发中具有重要的应用价值。
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
// 核函数：矩阵加法
__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    int rows = 1024; // 矩阵行数
    int cols = 1024; // 矩阵列数
    const size_t bytes = rows * cols * sizeof(float);
    // 主机内存分配
    std::vector<float> hostA(rows * cols, 1.0f);
    std::vector<float> hostB(rows * cols, 2.0f);
    std::vector<float> hostC(rows * cols);
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));
    // 动态调整线程块和网格大小
    int blockSize = 16;
    if (rows * cols > 1 << 20) {
        blockSize = 32;
    }
    dim3 threads(blockSize, blockSize);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    // 启动核函数
    matrixAdd<<<grid, threads>>>(deviceA, deviceB, deviceC, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
    // 打印部分结果
    std::cout << "Output: ";
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
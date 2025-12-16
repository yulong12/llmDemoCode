/**
 * @file example11_1.cu
 * @brief 演示如何实现基于多GPU的矩阵分块传输与计算调度。
 * @author zhangyulong 
 * @version 1.0
 * @date 2023-12-20
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// 定义矩阵分块大小
#define BLOCK_SIZE 16
// 矩阵乘法核函数
__global__ void matrixMultiplyKernel(
    const float *A, const float *B, float *C, int N, int subSize) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    if (row < subSize && col < N) {
        float sum=0;
        for (int k=0; k < N; ++k) {
            sum += A[row*N+k]*B[k*N+col];
        }
        C[row*N+col]=sum;
    }
}
// 多GPU矩阵分块乘法函数
void multiGPUMatrixMultiply(
    const float *A, const float *B, float *C, int N, int numGPUs) {
    int subSize=N / numGPUs;           // 每块矩阵大小
    size_t matrixSize=N*N*sizeof(float);

    std::vector<cudaStream_t> streams (numGPUs);
    std::vector<float *> d_A(numGPUs), d_B(numGPUs), d_C(numGPUs);

    // 初始化多GPU环境
    for (int i=0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        // 分配每个GPU的内存
        cudaMalloc((void **)&d_A[i], subSize*N*sizeof(float));
        cudaMalloc((void **)&d_B[i], matrixSize);
        cudaMalloc((void **)&d_C[i], subSize*N*sizeof(float));
    }

    // 复制数据到设备
    cudaMemcpyAsync(d_A[i], A+i*subSize*N, subSize*N*sizeof(float),
        cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(d_B[i], B, matrixSize, cudaMemcpyHostToDevice,
        streams[i]);
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N+BLOCK_SIZE-1) / BLOCK_SIZE,
        (subSize+BLOCK_SIZE-1) / BLOCK_SIZE);

    // 启动核函数
    for (int i=0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]
            >>>(d_A[i], d_B[i], d_C[i], N, subSize);
    }

    // 同步并将结果复制回主机
    for (int i=0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaMemcpyAsync(C+i*subSize*N, d_C[i], subSize*N*sizeof(float),
            cudaMemcpyDeviceToHost, streams[i]);
        cudaStreamSynchronize(streams[i]);

        // 释放设备内存
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }
}
int main() {
    int N=1024;                        // 矩阵大小
    int numGPUs=2;                     // 使用的GPU数量
    size_t matrixSize=N*N*sizeof(float);
    // 分配主机内存
    std::vector<float> A(N*N), B(N*N), C(N*N), C_ref(N*N);
    // 初始化矩阵A和矩阵B
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int i=0; i < N*N; ++i) {
        A[i]=distribution(generator);
        B[i]=distribution(generator);
    }

    // 使用多GPU进行矩阵乘法
    multiGPUMatrixMultiply(A.data(), B.data(), C.data(), N, numGPUs);
    // 验证结果
    for (int i=0; i < N; ++i) {

    for (int j=0; j < N; ++j) {
        float sum=0;
        for (int k=0; k < N; ++k) {
            sum += A[i*N+k]*B[k*N+j];
        }
        C_ref[i*N+j]=sum;
    }
    }
    // 检查结果是否正确
    bool correct=true;
    for (int i=0; i < N*N; ++i) {
        if (std::abs(C[i]-C_ref[i]) > 1e-3) {
            correct=false;
            break;
        }
    }
    if (correct) {
        std::cout << "矩阵乘法结果正确" << std::endl;
    } else {
        std::cout << "矩阵乘法结果错误" << std::endl;
    }
    return 0;
}


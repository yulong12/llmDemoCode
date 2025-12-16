/**
 * @file example11_2.cu
 * @brief 实现基于MPI和CUDA的多GPU矩阵分块计算和同步。假设代码中有多个节点，每个节点都配置了一个或多个GPU，每个进程负责一个节点的计算任务。请以矩阵加法为例，分配数据到不同GPU并完成计算，最后通过MPI进行结果汇总。
 * @author zhangyulong 
 * @version 1.0
 * @date 2023-12-20
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// CUDA核函数：矩阵加法
__global__ void matrixAddKernel(
    const float *A, const float *B, float *C, int rows, int cols) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (row < rows && col < cols) {
        C[row*cols+col]=A[row*cols+col]+B[row*cols+col];
    }
}
// GPU矩阵加法函数
void gpuMatrixAddition(const float *local_A, const float *local_B,
            float *local_C, int rows, int cols) {
    size_t size=rows*cols*sizeof(float);
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    // 复制数据到设备
    cudaMemcpy(d_A, local_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, local_B, size, cudaMemcpyHostToDevice);
    // 定义线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols+threadsPerBlock.x-1) / threadsPerBlock.x,
        (rows+threadsPerBlock.y-1) / threadsPerBlock.y);
    // 启动核函数
    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, rows, cols);
    // 同步并复制结果回主机
    cudaMemcpy(local_C, d_C, size, cudaMemcpyDeviceToHost);
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N=1024;                  // 矩阵行数
    const int M=1024;                  // 矩阵列数
    int local_rows=N / size;           // 每个进程处理的行数
    // 分配主机内存
    std::vector<float> local_A(local_rows*M);
    std::vector<float> local_B(local_rows*M);
    std::vector<float> local_C(local_rows*M);
    // 进程0初始化矩阵
    std::vector<float> A, B, C;
    if (rank == 0) {
        A.resize(N*M);
        B.resize(N*M);
        C.resize(N*M);
        // 初始化矩阵A和矩阵B
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        for (int i=0; i < N*M; ++i) {
            A[i]=distribution(generator);
            B[i]=distribution(generator);
        }
    }
    // 分发矩阵块到各进程
    MPI_Scatter(A.data(), local_rows*M, MPI_FLOAT, local_A.data(),
        local_rows*M, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data(), local_rows*M, MPI_FLOAT, local_B.data(),
        local_rows*M, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // GPU进行矩阵加法计算
    gpuMatrixAddition(local_A.data(), local_B.data(), local_C.data(),
        local_rows, M);
    // 收集结果到进程0
    MPI_Gather(local_C.data(), local_rows*M, MPI_FLOAT, C.data(),
        local_rows*M, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 进程0验证结果
    if (rank == 0) {
        bool correct=true;
        for (int i=0; i < N*M; ++i) {
            if (std::abs(C[i]-(A[i]+B[i])) > 1e-3) {
                correct=false;
                break;
            }
        }
        if (correct) {
            std::cout << "矩阵加法结果正确" << std::endl;
        } else {
            std::cout << "矩阵加法结果错误" << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}


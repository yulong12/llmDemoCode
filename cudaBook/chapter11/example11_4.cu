/**
 * @file example11_4.cu
 * @brief 在例11-3的基础上，使用多线程管理CPU任务，同时利用CUDA完成GPU部分的并行计算。
 * @author zhangyulong 
 * @version 1.0
 * @date 2023-12-20
 */
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>

// GPU核函数
__global__ void gpu_compute(const float* a, const float* b,
                 const float* w, float* result, int cols) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row < cols) {
        for (int j=0; j < cols; ++j) {
            result[row*cols+j]=a[row*cols+j]*
                w[row*cols+j]+b[row*cols+j]*(1-w[row*cols+j]);
        }
    }
}
// CPU计算函数
void cpu_compute(const std::vector<float>& a,
         const std::vector<float>& b,
         const std::vector<float>& w,
         std::vector<float>& result, int rows, int cols) {
    for (int i=0; i < rows; ++i) {
        for (int j=0; j < cols; ++j) {
            result[i*cols+j]=a[i*cols+j]*
                w[i*cols+j]+b[i*cols+j]*(1-w[i*cols+j]);
        }
    }
}
// 数据初始化函数
void initialize_matrix(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i=0; i < rows*cols; ++i) {
        matrix[i]=dis(gen);
    }
}

int main() {
    const int rows=10000;
    const int cols=10000;
    const int cpu_chunk_size=2000;

    // 初始化矩阵
    std::vector<float> matrix_a(rows*cols), matrix_b(rows*cols),
        weights(rows*cols);
    initialize_matrix(matrix_a, rows, cols);
    initialize_matrix(matrix_b, rows, cols);
    initialize_matrix(weights, rows, cols);

    // CPU部分
    std::vector<float> cpu_a(cpu_chunk_size*cols);
    std::vector<float> cpu_b(cpu_chunk_size*cols);
    std::vector<float> cpu_w(cpu_chunk_size*cols);
    std::vector<float> cpu_result(cpu_chunk_size*cols);

    // GPU部分
    int gpu_rows=rows-cpu_chunk_size;
    std::vector<float> gpu_a(gpu_rows*cols);
    std::vector<float> gpu_b(gpu_rows*cols);
    std::vector<float> gpu_w(gpu_rows*cols);

    // 划分数据
    std::copy(matrix_a.begin(), matrix_a.begin()+cpu_chunk_size*cols,
        cpu_a.begin());
    std::copy(matrix_b.begin(), matrix_b.begin()+cpu_chunk_size*cols,
        cpu_b.begin());
    std::copy(weights.begin(), weights.begin()+cpu_chunk_size*cols,
        cpu_w.begin());

    std::copy(matrix_a.begin()+cpu_chunk_size*cols, matrix_a.end(),
        gpu_a.begin());
    std::copy(matrix_b.begin()+cpu_chunk_size*cols, matrix_b.end(),
        gpu_b.begin());
    std::copy(weights.begin()+cpu_chunk_size*cols, weights.end(),
        gpu_w.begin());

    // 分配GPU内存
    float *d_a, *d_b, *d_w, *d_result;
    cudaMalloc((void**)&d_a, gpu_rows*cols*sizeof(float));
    cudaMalloc((void**)&d_b, gpu_rows*cols*sizeof(float));
    cudaMalloc((void**)&d_w, gpu_rows*cols*sizeof(float));
    cudaMalloc((void**)&d_result, gpu_rows*cols*sizeof(float));

    // 复制数据到GPU
    cudaMemcpy(d_a, gpu_a.data(), gpu_rows*cols*sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, gpu_b.data(), gpu_rows*cols*sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, gpu_w.data(), gpu_rows*cols*sizeof(float),
        cudaMemcpyHostToDevice);

    // 开始计时
    auto start_time=std::chrono::high_resolution_clock::now();

    // CPU线程
    std::thread cpu_thread(cpu_compute, std::ref(cpu_a), std::ref(cpu_b),
                std::ref(cpu_w), std::ref(cpu_result), cpu_chunk_size, cols);

    // GPU计算
    int block_size=256;
    int grid_size=(gpu_rows+block_size-1) / block_size;
    gpu_compute<<<grid_size, block_size>>>(d_a, d_b, d_w, d_result, cols);

    // 等待CPU线程
    cpu_thread.join();

    // 复制结果回CPU
    std::vector<float> gpu_result(gpu_rows*cols);
    cudaMemcpy(gpu_result.data(), d_result, gpu_rows*cols*sizeof(float),
        cudaMemcpyDeviceToHost);

    // 合并结果
    std::vector<float> final_result(rows*cols);
    std::copy(cpu_result.begin(), cpu_result.end(), final_result.begin());
    std::copy(gpu_result.begin(), gpu_result.end(), final_result.begin()+cpu_chunk_size*cols);

    auto end_time=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed=end_time-start_time;

    // 输出结果
    std::cout << "总计算时间： " << elapsed.count() << " 秒" << std::endl;
    std::cout << "结果矩阵前5行:" << std::endl;
    for (int i=0; i < 5; ++i) {
        for (int j=0; j < 5; ++j) {
            std::cout << final_result[i*cols+j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放GPU内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_w);
    cudaFree(d_result);

    return 0;
}


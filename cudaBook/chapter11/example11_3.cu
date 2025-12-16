/**
 * @file example11_3.cu
 * @brief 使用CUDA和标准C++实现CPU与GPU的协同计算，以大规模向量的加权求和为例。本示例将任务划分为CPU和GPU两部分，进行性能对比分析。
 * @author zhangyulong 
 * @version 1.0
 * @date 2023-12-20
 */
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cuda_runtime.h>
#include <chrono>

// GPU核函数
__global__ void gpu_weighted_sum_kernel(const float* a, const float* b,
                         const float* w, float* result, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < size) {
        result[idx]=a[idx]*w[idx]+b[idx]*(1.0f-w[idx]);
    }
}
// CPU加权求和函数
void cpu_weighted_sum(const std::vector<float>& a,
             const std::vector<float>& b,
             const std::vector<float>& w, std::vector<float>& result) {
    for (size_t i=0; i < a.size(); ++i) {
        result[i]=a[i]*w[i]+b[i]*(1.0f-w[i]);
    }
}
// 随机数生成
std::vector<float> generate_random_vector(int size) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : vec) {
        val=dist(gen);
    }
    return vec;
}

int main() {
    int vector_size=10'000'000;
    int cpu_chunk_size=vector_size / 10;

    // 初始化数据
    std::vector<float> vector_a=generate_random_vector(vector_size);
    std::vector<float> vector_b=generate_random_vector(vector_size);
    std::vector<float> weights=generate_random_vector(vector_size);

    // CPU部分数据
    std::vector<float> cpu_a(vector_a.begin(),
                 vector_a.begin()+cpu_chunk_size);
    std::vector<float> cpu_b(vector_b.begin(),
                 vector_b.begin()+cpu_chunk_size);
    std::vector<float> cpu_w(weights.begin(),
                 weights.begin()+cpu_chunk_size);
    std::vector<float> cpu_result(cpu_chunk_size);

    // GPU部分数据
    std::vector<float> gpu_a(vector_a.begin()+cpu_chunk_size,
                 vector_a.end());
    std::vector<float> gpu_b(vector_b.begin()+cpu_chunk_size,
                 vector_b.end());
    std::vector<float> gpu_w(weights.begin()+cpu_chunk_size,
                 weights.end());
    std::vector<float> gpu_result(gpu_chunk_size);

    int gpu_chunk_size=vector_size-cpu_chunk_size;
    std::vector<float> gpu_result(gpu_chunk_size);

    float *d_a, *d_b, *d_w, *d_result;
    cudaMalloc(&d_a, gpu_chunk_size*sizeof(float));
    cudaMalloc(&d_b, gpu_chunk_size*sizeof(float));
    cudaMalloc(&d_w, gpu_chunk_size*sizeof(float));
    cudaMalloc(&d_result, gpu_chunk_size*sizeof(float));

    // 传输数据到GPU
    cudaMemcpy(d_a, gpu_a.data(), gpu_chunk_size*sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, gpu_b.data(), gpu_chunk_size*sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, gpu_w.data(), gpu_chunk_size*sizeof(float),
        cudaMemcpyHostToDevice);

    // CPU计算
    auto start_cpu=std::chrono::high_resolution_clock::now();
    cpu_weighted_sum(cpu_a, cpu_b, cpu_w, cpu_result);
    auto end_cpu=std::chrono::high_resolution_clock::now();

    // GPU计算
    dim3 threadsPerBlock(256);
    dim3 numBlocks(
        (gpu_chunk_size+threadsPerBlock.x-1) / threadsPerBlock.x);
    auto start_gpu=std::chrono::high_resolution_clock::now();
    gpu_weighted_sum_kernel<<<numBlocks, threadsPerBlock>>>(
        d_a, d_b, d_w, d_result, gpu_chunk_size);
    cudaDeviceSynchronize();
    auto end_gpu=std::chrono::high_resolution_clock::now();

    // 传输结果回主机
    cudaMemcpy(gpu_result.data(), d_result, gpu_chunk_size*sizeof(float),
        cudaMemcpyDeviceToHost);

    // 合并结果
    std::vector<float> final_result;
    final_result.insert(final_result.end(), cpu_result.begin(),
                 cpu_result.end());
    final_result.insert(final_result.end(), gpu_result.begin(),
                 gpu_result.end());

    // 性能输出
    auto cpu_duration=std::chrono::duration_cast<
        std::chrono::microseconds>(end_cpu-start_cpu).count();
    auto gpu_duration=std::chrono::duration_cast<
        std::chrono::microseconds>(end_gpu-start_gpu).count();
    std::cout << "CPU计算时间： " << cpu_duration / 1e6 << " 秒\n";
    std::cout << "GPU计算时间： " << gpu_duration / 1e6 << " 秒\n";
    std::cout << "总任务结果前10项： ";
    for (int i=0; i < 10; ++i) {
        std::cout << final_result[i] << " ";
    }
    std::cout << std::endl;

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_w);
    cudaFree(d_result);

    return 0;
}
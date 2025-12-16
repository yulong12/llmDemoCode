/**
 * @file example11_7.cu
 * @brief 实现一个动态任务分配和负载均衡的实例，通过模拟多GPU执行不同复杂度的任务，演示如何动态调整任务分配策略以实现负载均衡。
 * @author zhangyulong (zhangyulong@ict.ac.cn)
 * @version 1.0
 * @date 2023-12-20
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
// 核函数模拟不同计算任务
__global__ void computeTask(int task_id, int *output, int workload) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < workload) {
        output[idx] = task_id * idx; // 模拟计算任务
    }
}
// 动态分配任务并调度到不同GPU
void dynamicTaskAllocation(const std::vector<int>& workloads, int num_gpus) {
    int *device_outputs[16]; // 假设最多支持16个GPU
    cudaStream_t streams[16];
    
    // 初始化设备和流
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&device_outputs[i], workloads[i] * sizeof(int));
        cudaStreamCreate(&streams[i]);
    }
    
    // 动态分配任务
    for (int i = 0; i < workloads.size(); ++i) {
        int device_id = i % num_gpus;
        cudaSetDevice(device_id);
        computeTask<<<(workloads[i] + 255) / 256, 256, 0, streams[device_id]>>>(i, device_outputs[device_id], workloads[i]);
    }
    // 同步设备
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    // 释放资源
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaFree(device_outputs[i]);
        cudaStreamDestroy(streams[i]);
    }
}
int main() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::cout << "Detected " << num_gpus << " GPUs." << std::endl;
    std::vector<int> workloads = {1000, 2000, 3000, 1500, 2500}; // 模拟不同任务的计算量
    std::cout << "Starting dynamic task allocation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    dynamicTaskAllocation(workloads, num_gpus);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Task allocation and execution completed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms." << std::endl;
    return 0;
}


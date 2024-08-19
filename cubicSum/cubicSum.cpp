// cubicSum.cpp : 立方之和。
//

#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <future>
#include <vector>

// 计算数组中所有元素的立方和
double cubicSum(const double* array, long int length) {
    double sum = 0.0;

    for (int i = 0; i < length; i++) {
        sum += array[i] * array[i] * array[i];
    }
    return sum;
}

// 计算数组中所有元素的立方和
double cubicSum_MT(const double* array, long int length) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    //reduction(+:sum) 来告诉 OpenMP 运行时系统如何处理 sum 变量。
    //+ 表示所有线程计算的结果将通过加法合并到 sum 变量中。
    for (int i = 0; i < length; i++) {
        sum += array[i] * array[i] * array[i];
    }
    return sum;
}

// 计算数组中所有元素的立方和
double cubicSum_MTv2(const double* array, long int length) {
    double sum = 0.0;
#pragma omp parallel num_threads(std::thread::hardware_concurrency())
    {
        size_t thread_id = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();

        // Calculate the start and end indices for this thread
        size_t start = (length / num_threads) * thread_id;
        size_t end = (thread_id != num_threads - 1) ?
            (length / num_threads) * (thread_id + 1) :
            length;

        double local_sum = 0.0;

        // Compute the local sum for this thread
        for (size_t i = start; i < end; i++) {
            local_sum += array[i] * array[i] * array[i];
        }

        // Accumulate the local sum into the global sum
#pragma omp atomic
        sum += local_sum;
    }
    return sum;
}



int main() {
    const long int length = 300000000; // 数组长度
    double* array = new double[length];

    // 初始化数组
    for (int i = 0; i < length; i++) {
        array[i] = i;
    }

    // 计算并行前的时间
    auto start1 = std::chrono::high_resolution_clock::now();
    double serial_sum = cubicSum(array, length);
    auto end1 = std::chrono::high_resolution_clock::now();

    // 计算并行化后的时间
    auto start2 = std::chrono::high_resolution_clock::now();
    double parallel_sum = cubicSum_MT(array, length);
    auto end2 = std::chrono::high_resolution_clock::now();

    // 计算并行化后的时间
    auto start3 = std::chrono::high_resolution_clock::now();
    double parallel_sum_v2 = cubicSum_MTv2(array, length);
    auto end3 = std::chrono::high_resolution_clock::now();

    // 输出结果和耗时
    std::cout << "Serial sum: " << serial_sum << std::endl;
    std::cout << "Parallel sum: " << parallel_sum << std::endl;
    std::cout << "Parallel sum v2: " << parallel_sum_v2 << std::endl;
    std::cout << "Time for serial: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;
    std::cout << "Time for parallel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;
    std::cout<< "Time for parallel v2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << "ms" << std::endl;
    delete[] array;
    return 0;
}


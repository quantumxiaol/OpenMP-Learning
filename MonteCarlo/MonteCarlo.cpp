// MonteCarlo.cpp：蒙特卡洛方法
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>
#include <vector>

// 计算PI的函数
double monteCarloPi(long numSamples) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    long inside = 0;

    for (long i = 0; i < numSamples; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);
        if (x * x + y * y <= 1.0) {
            ++inside;
        }
    }

    return 4.0 * inside / numSamples;
}

// 定义一个原子加法函数
//也可以直接使用 std::atomic<long> 类型来替代 long
void atomic_add(volatile long* target, long value) {
#pragma omp atomic
    * target += value;
}

// 使用 OpenMP 并行化的蒙特卡洛方法计算 π
double parallelMonteCarloPi(long numSamples) {
    std::atomic<long> totalInside(0);

#pragma omp parallel
    {
        std::default_random_engine generator(omp_get_thread_num()); // 使用不同的种子
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        long localInside = 0;
        long chunkSize = numSamples / omp_get_num_threads(); // 每个线程处理的数据量
        long start = omp_get_thread_num() * chunkSize;
        long end = start + chunkSize;

        // 确保最后一个线程处理所有剩余的数据
        if (omp_get_thread_num() == omp_get_num_threads() - 1) {
            end = numSamples;
        }

        for (long i = start; i < end; ++i) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x * x + y * y <= 1.0) {
                ++localInside;
            }
        }

        // 在退出临界区之前更新 totalInside
#pragma omp critical
        {
            totalInside += localInside;
        }
    }

    return 4.0 * totalInside / numSamples;
}

int main() {
    const long numSamples = 10000000000; // 可以调整样本数量来提高精度

    // 不使用 OpenMP 的情况
    auto start = std::chrono::high_resolution_clock::now();
    double pi = monteCarloPi(numSamples);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Non-parallel Monte Carlo Pi: " << pi << "\nTime taken: " << elapsed.count() << " seconds\n";

    // 使用 OpenMP 的情况
    start = std::chrono::high_resolution_clock::now();
    double parallelPi = parallelMonteCarloPi(numSamples);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Parallel Monte Carlo Pi: " << parallelPi << "\nTime taken: " << elapsed.count() << " seconds\n";

    return 0;
}
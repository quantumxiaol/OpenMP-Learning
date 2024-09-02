// MonteCarlo.cpp�����ؿ��巽��
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>
#include <vector>

// ����PI�ĺ���
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

// ����һ��ԭ�Ӽӷ�����
//Ҳ����ֱ��ʹ�� std::atomic<long> ��������� long
void atomic_add(volatile long* target, long value) {
#pragma omp atomic
    * target += value;
}

// ʹ�� OpenMP ���л������ؿ��巽������ ��
double parallelMonteCarloPi(long numSamples) {
    std::atomic<long> totalInside(0);

#pragma omp parallel
    {
        std::default_random_engine generator(omp_get_thread_num()); // ʹ�ò�ͬ������
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        long localInside = 0;
        long chunkSize = numSamples / omp_get_num_threads(); // ÿ���̴߳����������
        long start = omp_get_thread_num() * chunkSize;
        long end = start + chunkSize;

        // ȷ�����һ���̴߳�������ʣ�������
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

        // ���˳��ٽ���֮ǰ���� totalInside
#pragma omp critical
        {
            totalInside += localInside;
        }
    }

    return 4.0 * totalInside / numSamples;
}

int main() {
    const long numSamples = 10000000000; // ���Ե���������������߾���

    // ��ʹ�� OpenMP �����
    auto start = std::chrono::high_resolution_clock::now();
    double pi = monteCarloPi(numSamples);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Non-parallel Monte Carlo Pi: " << pi << "\nTime taken: " << elapsed.count() << " seconds\n";

    // ʹ�� OpenMP �����
    start = std::chrono::high_resolution_clock::now();
    double parallelPi = parallelMonteCarloPi(numSamples);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Parallel Monte Carlo Pi: " << parallelPi << "\nTime taken: " << elapsed.count() << " seconds\n";

    return 0;
}
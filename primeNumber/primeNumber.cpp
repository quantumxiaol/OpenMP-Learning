//primeNumber.cpp 质数筛选
//给定数字n，输出从1到n之间的所有质数

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// 定义筛选质数的函数
void naive_prime_sieve(long int n, std::vector<long int>& PrimeNumber);
void optimized_naive_prime_sieve(long int n, std::vector<long int>& PrimeNumber);
void omp_prime_sieve(long int n, std::vector<long int>& PrimeNumber);
void eratosthenes_sieve(long int n, std::vector<long int>& PrimeNumber);
void linear_sieve(long int n, std::vector<long int>& PrimeNumber);

void naive_prime_sieve(long int n, std::vector<long int>& PrimeNumber) {
    for (long int i = 2; i <= n; ++i) {
        bool is_prime = true;
        for (long int j = 2; j < i; ++j) {
            if (i % j == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            PrimeNumber.push_back(i);
        }
    }
}

void optimized_naive_prime_sieve(long int n, std::vector<long int>& PrimeNumber) {
    for (long int i = 2; i <= n; ++i) {
        bool is_prime = true;
        for (long int j = 2; j * j <= i; ++j) {
            if (i % j == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            PrimeNumber.push_back(i);
        }
    }
}

void omp_prime_sieve(long int n, std::vector<long int>& PrimeNumber) {
#pragma omp parallel
    {
        std::vector<long int> threadLocalPrimes;
#pragma omp for
        for (long int i = 2; i <= n; ++i) {
            bool is_prime = true;
            for (long int j = 2; j * j <= i; ++j) {
                if (i % j == 0) {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime) {
                threadLocalPrimes.push_back(i);
            }
        }

#pragma omp critical
        for (long int prime : threadLocalPrimes) {
            PrimeNumber.push_back(prime);
        }
    }
}

void eratosthenes_sieve(long int n, std::vector<long int>& PrimeNumber) {
    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (long int i = 2; i * i <= n; ++i) {
        if (is_prime[i]) {
            for (long int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    for (long int i = 2; i <= n; ++i) {
        if (is_prime[i]) {
            PrimeNumber.push_back(i);
        }
    }
}

void linear_sieve(long int n, std::vector<long int>& PrimeNumber) {
    std::vector<long int> min_prime(n + 1, 0);
    for (long int i = 2; i <= n; ++i) {
        if (min_prime[i] == 0) {
            min_prime[i] = i;
            PrimeNumber.push_back(i);
        }
        for (long int j = 0; j < PrimeNumber.size() && i * PrimeNumber[j] <= n; ++j) {
            min_prime[i * PrimeNumber[j]] = PrimeNumber[j];
            if (i % PrimeNumber[j] == 0) {
                break;
            }
        }
    }
}



int main() {
    long int n = 100000;  // 1到n之间的质数
    std::vector<long int> PrimeNumber1;
    std::vector<long int> PrimeNumber2;
    std::vector<long int> PrimeNumber3;
    std::vector<long int> PrimeNumber4;
    std::vector<long int> PrimeNumber5;

        // 比较五种方法的结果和运行时间
        auto start1 = std::chrono::steady_clock::now();
        naive_prime_sieve(n, PrimeNumber1);
        auto end1 = std::chrono::steady_clock::now();
        std::cout << "Naive Prime Sieve: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;

        auto start2 = std::chrono::steady_clock::now();
        optimized_naive_prime_sieve(n, PrimeNumber2);
        auto end2 = std::chrono::steady_clock::now();
        std::cout << "Optimized Naive Prime Sieve: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;

        auto start3 = std::chrono::steady_clock::now();
        omp_prime_sieve(n, PrimeNumber3);
        auto end3 = std::chrono::steady_clock::now();
        std::cout << "OpenMP Prime Sieve: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << "ms" << std::endl;

        auto start4 = std::chrono::steady_clock::now();
        eratosthenes_sieve(n, PrimeNumber4);
        auto end4 = std::chrono::steady_clock::now();
        std::cout << "Eratosthenes Sieve: " << std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4).count() << "ms" << std::endl;

        auto start5 = std::chrono::steady_clock::now();
        linear_sieve(n, PrimeNumber5);
        auto end5 = std::chrono::steady_clock::now();
        std::cout << "Linear Sieve: " << std::chrono::duration_cast<std::chrono::milliseconds>(end5 - start5).count() << "ms" << std::endl;

        //输出前m个质数
    long int m = 100;
    std::cout << "Prime Sieve: ";
    for (long int i = 0; i < m; ++i) {
		std::cout << PrimeNumber2[i] << " ";
        if(i % 10 == 9)
			std::cout << std::endl;
	}

    return 0;
}
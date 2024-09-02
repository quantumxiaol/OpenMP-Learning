//polynomialMultiplication.cpp : 多项式乘法
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <omp.h>
#include <chrono>
//#include <numbers>

// 快速傅里叶变换
void fft(std::vector<std::complex<double>>& x, bool invert) {
    int N = x.size();

    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            std::swap(x[i], x[j]);
    }

    for (int len = 2; len <= N; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < N; i += len) {
            std::complex<double> w(1);
            for (int j = 0; j < len / 2; j++) {
                std::complex<double> u = x[i + j], v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (int i = 0; i < N; i++)
            x[i] /= N;
    }
}

// 多项式乘法
void polynomialMultiplication(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& MUL) {
    int n = 1;
    while (n < A.size() + B.size()) {
        n <<= 1;
    }

    std::vector<std::complex<double>> fa(n), fb(n);
    for (size_t i = 0; i < A.size(); ++i) {
        fa[i] = std::complex<double>(A[i], 0);
    }
    for (size_t i = 0; i < B.size(); ++i) {
        fb[i] = std::complex<double>(B[i], 0);
    }

    fft(fa, false);
    fft(fb, false);

    for (int i = 0; i < n; i++) {
        fa[i] *= fb[i];
    }

    fft(fa, true);

    MUL.resize(n);
    for (int i = 0; i < n; i++) {
        MUL[i] = fa[i].real();
    }
}

// 朴素多项式乘法
void naivePolynomialMultiplication(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& MUL) {
    MUL.resize(A.size() + B.size() - 1);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B.size(); ++j) {
            MUL[i + j] += A[i] * B[j];
        }
    }
}

// 使用 OpenMP 的朴素多项式乘法
void openmpNaivePolynomialMultiplication(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& MUL) {
    MUL.resize(A.size() + B.size() - 1);
    int i = 0;
#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B.size(); ++j) {
            MUL[i + j] += A[i] * B[j];
        }
    }
}

// 结合 OpenMP 和 FFT 优化的多项式乘法
void openmpPolynomialMultiplication(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& MUL) {
    int n = 1;
    while (n < A.size() + B.size()) {
        n <<= 1;
    }
    int i = 0;
    std::vector<std::complex<double>> fa(n), fb(n);
#pragma omp parallel for schedule(static)
    for (i = 0; i < A.size(); ++i) {
        fa[i] = std::complex<double>(A[i], 0);
    }
#pragma omp parallel for schedule(static)
    for (i = 0; i < B.size(); ++i) {
        fb[i] = std::complex<double>(B[i], 0);
    }

    fft(fa, false);
    fft(fb, false);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        fa[i] *= fb[i];
    }

    fft(fa, true);

    MUL.resize(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        MUL[i] = fa[i].real();
    }
}

// 打印多项式的前len项
void PrintV(const std::vector<double>& A, int len) {
    // 遍历多项式的前len项
    for (int i = 0; i < len && i < A.size(); ++i) {
        // 如果系数不为0，则打印该项
        if (A[i] != 0) {
            if (i > 0) std::cout << "+"; // 除了第一项外，其余项前面都加'+'号
            if (A[i] != 1 || i == 0) std::cout << A[i]; // 如果系数不是1或者当前项是常数项，则打印系数
            if (i == 1) std::cout << "x"; // 当前项是一次项
            else if (i > 1) std::cout << "x^" << i; // 当前项是高于一次的项
        }
    }
    std::cout << std::endl;
}

int main() {
    // 初始化随机数生成器
    std::mt19937 gen(std::time(nullptr));
    std::uniform_real_distribution<> dis(-100, 100);

    // 长度为10000的多项式
    int length = 10000;
    std::vector<double> A(length);
    std::vector<double> B(length);

    // 为多项式随机赋值
    for (int i = 0; i < length; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }
    PrintV(A, 10);
    PrintV(B, 10);

    std::cout << "Comparing algorithms:" << std::endl;

    // 比较四种算法的结果和运行时间
    {
        std::vector<double> result(length + length - 1);
        auto start = std::chrono::high_resolution_clock::now();
        naivePolynomialMultiplication(A, B, result);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Naive Polynomial Multiplication took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds." << std::endl;
        PrintV(result, 10);
    }

    {
        std::vector<double> result(length + length - 1);
        auto start = std::chrono::high_resolution_clock::now();
        openmpNaivePolynomialMultiplication(A, B, result);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "OpenMP Naive Polynomial Multiplication took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds." << std::endl;
        PrintV(result, 10);
    }

    {
        std::vector<double> result(length + length - 1);
        auto start = std::chrono::high_resolution_clock::now();
        polynomialMultiplication(A, B, result);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "FFT Optimized Polynomial Multiplication took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds." << std::endl;
        PrintV(result, 10);
    }

    {
        std::vector<double> result(length + length - 1);
        auto start = std::chrono::high_resolution_clock::now();
        openmpPolynomialMultiplication(A, B, result);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "OpenMP and FFT Optimized Polynomial Multiplication took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds." << std::endl;
        PrintV(result, 10);
    }

    return 0;
}
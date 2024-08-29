//matrixMultiplication.cpp：矩阵乘法

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <Eigen/Dense> // Eigen 线性代数库
#include <random>

void matrixMultiplySerial(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // 初始化结果矩阵
    C.resize(rowsA, std::vector<double>(colsB, 0));

    // 矩阵乘法
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrixMultiplyParallel(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // 初始化结果矩阵
    C.resize(rowsA, std::vector<double>(colsB, 0));

    // 并行化矩阵乘法
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void matrixMultiplyEigen(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    C = A * B;
}
// 优化后的并行矩阵乘法
void matrixMultiplyParallelOptimized(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // 初始化结果矩阵
    C.resize(rowsA, std::vector<double>(colsB, 0));

    // 并行化矩阵乘法
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
//simd指示编译器尝试对一个循环进行向量化执行，这对于一些简单的循环特别有用
//所有的线程对 sum 的贡献会被加在一起
#pragma omp simd reduction(+:sum)
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void blockMatrixMultiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int blockSize = 64) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // 初始化结果矩阵
    C.resize(rowsA, std::vector<double>(colsB, 0));

    // 并行化矩阵乘法
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsA; i += blockSize) {
        for (int j = 0; j < colsB; j += blockSize) {
            for (int k = 0; k < colsA; k += blockSize) {
                for (int ii = 0; ii < blockSize && (i + ii) < rowsA; ++ii) {
                    for (int jj = 0; jj < blockSize && (j + jj) < colsB; ++jj) {
                        double sum = 0.0;
                        for (int kk = 0; kk < blockSize && (k + kk) < colsA; ++kk) {
                            sum += A[i + ii][k + kk] * B[k + kk][j + jj];
                        }
                        C[i + ii][j + jj] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    // 设置矩阵的大小
    int M = 100; // 行数
    int N = 100; // 列数
    int P = 100; // 第二个矩阵的列数

    // 随机生成矩阵 A 和 B
    std::vector<std::vector<double>> A(M, std::vector<double>(N, 0));
    std::vector<std::vector<double>> B(N, std::vector<double>(P, 0));
    std::vector<std::vector<double>> C_serial(M, std::vector<double>(P, 0));
    std::vector<std::vector<double>> C_parallel(M, std::vector<double>(P, 0));
    std::vector<std::vector<double>> C_block(M, std::vector<double>(P, 0));

    Eigen::MatrixXd A_eigen(M, N);
    Eigen::MatrixXd B_eigen(N, P);
    Eigen::MatrixXd C_eigen(M, P);

    std::default_random_engine engine;
    std::uniform_real_distribution<double> distribution(-10.0, 10.0);

    // 初始化矩阵 A 和 B
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = distribution(engine);
            A_eigen(i, j) = A[i][j];
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            B[i][j] = distribution(engine);
            B_eigen(i, j) = B[i][j];
        }
    }

    // 串行矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplySerial(A, B, C_serial);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Serial matrix multiplication took " << duration << " microseconds." << std::endl;

    // 并行矩阵乘法
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyParallelOptimized(A, B, C_parallel);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Parallel matrix multiplication took " << duration << " microseconds." << std::endl;

    // 块矩阵乘法
    start = std::chrono::high_resolution_clock::now();
    blockMatrixMultiply(A, B, C_block, 64); // 假设每个块的大小为 64x64
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Block matrix multiplication took " << duration << " microseconds." << std::endl;

    // 使用 Eigen 库的矩阵乘法
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyEigen(A_eigen, B_eigen, C_eigen);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Eigen library matrix multiplication took " << duration << " microseconds." << std::endl;

    return 0;
}
//matrixMultiplication.cpp：矩阵乘法
//
//
// MacOS
// /opt/homebrew/opt/llvm/bin/clang++ \
  -std=c++17 -fopenmp -O2 \
  -I/opt/homebrew/include/eigen3 \
  matrixMultiplication/matrixMultiplication.cpp -o output/matrixMultiplication \
  $(pkg-config --cflags --libs opencv4 pcl_common pcl_io)
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

int main(int argc, char* argv[]) {
    // 检查是否提供了足够的参数
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N P" << std::endl;
        return 1;
    }

    // 设置矩阵的大小
    // 从命令行参数读取 M, N, P
    int M = std::atoi(argv[1]);// 行数
    int N = std::atoi(argv[2]);// 列数
    int P = std::atoi(argv[3]);// 第二个矩阵的列数

    // 检查输入合法性
    if (M <= 0 || N <= 0 || P <= 0) {
        std::cerr << "Error: M, N, P must be positive integers." << std::endl;
        return 1;
    }
 

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
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyParallelOptimized(A, B, C_parallel);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Parallel matrix multiplication took " << duration << " microseconds." << std::endl;

    // 块矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    blockMatrixMultiply(A, B, C_block, 64); // 假设每个块的大小为 64x64
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Block matrix multiplication took " << duration << " microseconds." << std::endl;

    // 使用 Eigen 库的矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyEigen(A_eigen, B_eigen, C_eigen);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Eigen library matrix multiplication took " << duration << " microseconds." << std::endl;

    return 0;
}
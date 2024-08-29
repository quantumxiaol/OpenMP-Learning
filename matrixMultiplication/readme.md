# 矩阵乘法

串行矩阵乘法：实现了基本的矩阵乘法算法。

并行矩阵乘法：使用 OpenMP 并行化了外层循环，每个线程处理矩阵的不同行。

Eigen 库：使用 Eigen 库实现矩阵乘法，这是一种高性能的 C++ 线性代数库。

## 测试与比较

在 main 函数中，随机生成了两个矩阵 A 和 B，并通过 std::chrono 库测量了三种实现方式的执行时间。

### C++ for循环

矩阵乘法是一个基本的数学运算，两个矩阵A(m,n)*B(n,p)=C(m,p)，C的第i行第j列的元素是A第i行和B第j列对应的元素相乘再求和得到的结果。

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

### C++ 并行化矩阵乘法

VS需要在C/C++ - 命令行中开启-openmp:experimental以使用Simd。

  #pragma omp parallel for：告诉编译器将一个循环中的迭代并行化，使得每个线程可以处理循环的一部分迭代。
  
  schedule(static)：指定调度策略为静态，意味着在运行前就决定每个线程应该处理哪些迭代。
  
  #pragma omp simd：指示编译器尝试对一个循环进行向量化执行，这对于一些简单的循环特别有用。
  
  reduction(+:sum)：表示在并行区域内对 sum 变量的更新操作需要被正确地合并，即所有的线程对 sum 的贡献会被加在一起。

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

### C++ 分块乘法

分块乘法是一种将大矩阵分解成小块，然后分别处理的方法。这种方法有助于更好地利用缓存，减少内存访问延迟，并且可以更容易地管理并行任务。
在 blockMatrixMultiply 函数中，矩阵被分割成大小为 blockSize 的块，每个块的乘法可以在不同的线程中并行完成。

使用了 collapse(2) 指令来并行化两个嵌套循环。这使得整个矩阵被分割成多个小块，并且每个小块的乘法可以独立地在不同的线程中完成。

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

### C++ Eigen 线性代数库

Eigen 是一个高效的线性代数库，使用 Eigen 进行矩阵乘法非常简单，只需调用 * 运算符即可。

    C=A*B;

运行结果
Serial matrix multiplication took 1821997 microseconds.
Parallel matrix multiplication took 487250 microseconds.
Block matrix multiplication took 361976 microseconds.
Eigen library matrix multiplication took 166218 microseconds.

可见并行化可以提高速度，分块计算可以进一步提高速度。当然Eigen 还是最快的。

顺便对比一下python、NumPy和PyTorch的表现。

### Python PyTorch 

    C = torch.matmul(A, B)

### Python NumPy
    C_numpy = np.dot(A, B)

### Python for循环

    def matrix_multiply_python(A, B):
        m, n = A.shape
        n, p = B.shape
        
        C = np.zeros((m, p))
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        
        return C

在矩阵为100时，纯python也能算出来，numpy还要快。

PyTorch CUDA matrix multiplication took 376465.80 microseconds.

Numpy matrix multiplication took 1994.37 microseconds.

Python matrix multiplication took 888069.15 microseconds.

Serial matrix multiplication took 1150 microseconds.

Parallel matrix multiplication took 532 microseconds.

Block matrix multiplication took 1830 microseconds.

Eigen library matrix multiplication took 374 microseconds.


在矩阵为1000时，对比C++的四种方式

PyTorch CUDA matrix multiplication took 383928.06 microseconds.

Numpy matrix multiplication took 6979.70 microseconds.

Serial matrix multiplication took 1821997 microseconds.

Parallel matrix multiplication took 487250 microseconds.

Block matrix multiplication took 361976 microseconds.

Eigen library matrix multiplication took 166218 microseconds.


在矩阵为10000时，torch就更快了

PyTorch CUDA matrix multiplication took 462138.41 microseconds.

Numpy matrix multiplication took 5257691.38 microseconds.

此时C++ 也只有Eigen能够算了，但Eigen居然不如NumPy快

Block matrix multiplication took 299894005 microseconds.

Eigen library matrix multiplication took 38899864 microseconds.

结果：

| 矩阵尺寸 | Python   | NumPy            | PyTorch          | C++  | OMP           | BLOCKOMP         | Eigen         |
|----------|----------|------------------|------------------|------|---------------|------------------|---------------|
| 100      | 888069.15| 1994.37          | 376465.8         | 1150 | 532           | 1830             | 374           |
| 数量级   | 6        | 4                | 6                | 4    | 3             | 4                | 3             |
| 1000     | -        | 6979.7           | 383928.06        | 1821997 | 487250        | 361976           | 166218        |
| 数量级   |          | 4                | 6                | 7    | 6             | 6                | 6             |
| 10000    | -        | 5257691.38       | 462138.41        | -    | -             | 299894005        | 38899864      |
| 数量级   |          | 7                | 6                |      |               | 9                | 8             |


从表中可以看出，随着矩阵尺寸的增加，CUDA版本的PyTorch在大规模数据集上表现出了明显的优势。

对于较小的矩阵（如100x100），某些CPU优化的方法（如多线程并行计算和平铺矩阵乘法）可以提供很好的性能。

而对于较大的矩阵（如10000x10000），只有Eigen库能够在合理的时间内完成计算，尽管其性能不如Numpy或PyTorch CUDA。

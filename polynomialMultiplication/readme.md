# 多项式乘法

现有多项式

    A = A0 + A1*x^1+ A2*x^2 +···+ An*x^n
    
    B = B0 + B1*x^1+ B2*x^2 +···+ Bm*x^m

两个多项式相乘得到C，C的长度为m+n。

现在计算两个多项式的乘积。

朴素多项式乘法是最简单的方法，它直接通过双重循环逐项相乘并累加结果。C[ i ]=Ak*Bi-k（k从0到i）。

    void naivePolynomialMultiplication(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& MUL) {
        MUL.resize(A.size() + B.size() - 1);
        for (size_t i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < B.size(); ++j) {
                MUL[i + j] += A[i] * B[j];
            }
        }
    }

使用OpenMP并行化朴素多项式乘法中的双重循环，通过将外层循环并行化来加速计算。

    #pragma omp parallel for 指令将循环分配给多个线程，使得每个线程处理循环的一部分，从而提高计算效率。
    #pragma omp parallel for schedule(dynamic)
        for (i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < B.size(); ++j) {
                MUL[i + j] += A[i] * B[j];
            }
    }

快速傅里叶变换可以用于多项式乘法中，这是一种有效的方法，特别是在处理大系数或多高次项的情况下。传统的多项式乘法需要O(N^2)的操作，其中N是多项式的最高次幂。使用FFT可以将多项式的系数表示转换为它们的点值表示（在一组特定点上的值），然后在频域中执行乘法，最后通过逆FFT转换回系数表示。

以下是使用FFT进行多项式乘法的基本步骤：

准备：设A(x)和B(x)为两个多项式，首先确保它们的系数向量长度相同且为2的幂，这可以通过在末尾添加零来完成（称为补零）。

前向FFT：分别对A(x)和B(x)的系数应用FFT，得到它们在某些复数根上的点值表示。

点乘：由于多项式的乘法在点值表示下转化为对应点的值相乘，因此只需简单地将A(x)和B(x)在相同点上的值相乘，得到结果多项式C(x)在这些点上的值。

逆FFT：对C(x)的点值表示应用逆FFT，恢复出C(x)的系数表示

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

多项式乘法

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

将多项式的FFT和IFFT过程中的循环分配给多个线程，可以进一步优化效率。


-75.2478+-29.2987x+-24.2463x^2+98.3284x^3+52.6919x^4+-53.4377x^5+-86.8263x^6+-78.9493x^7+73.1755x^8+-38.8597x^9

47.2873+-29.9648x+96.7859x^2+-21.7494x^3+18.0662x^4+-67.914x^5+-46.3064x^6+-49.4897x^7+-17.9707x^8+88.9099x^9

Comparing algorithms:

Naive Polynomial Multiplication took 90325 microseconds.

-3558.27+869.331x+-7551.54x^2+4177.12x^3+-3523.65x^4+10519.4x^5+5492.93x^6+1054.19x^7+-3216.28x^8+-23844x^9

OpenMP Naive Polynomial Multiplication took 24464 microseconds.

-3558.27+869.331x+-7551.54x^2+4177.12x^3+-3523.65x^4+10519.4x^5+5492.93x^6+1054.19x^7+-3216.28x^8+-23844x^9

FFT Optimized Polynomial Multiplication took 3617 microseconds.

-3558.27+869.331x+-7551.54x^2+4177.12x^3+-3523.65x^4+10519.4x^5+5492.93x^6+1054.19x^7+-3216.28x^8+-23844x^9

OpenMP and FFT Optimized Polynomial Multiplication took 3302 microseconds.

-3558.27+869.331x+-7551.54x^2+4177.12x^3+-3523.65x^4+10519.4x^5+5492.93x^6+1054.19x^7+-3216.28x^8+-23844x^9

可见朴素多项式算法耗时90325ms，使用OMP并行化后耗时24464ms，而仅使用FFT优化后耗时即可缩短至3617ms，进一步使用OMP并行化也可优化至3302ms，但效果不明显，因为优化的都是单层循环。



使用Python来对比，PyTorch和NumPy都有FFT。

朴素多项式乘法 (Naive Polynomial Multiplication)

    def naive_polynomial_multiplication(A, B):
        n = len(A) + len(B) - 1
        C = np.zeros(n)
        for i in range(len(A)):
            for j in range(len(B)):
                C[i + j] += A[i] * B[j]
        return C

使用 NumPy 的 FFT 优化的多项式乘法 (FFT Optimized Polynomial Multiplication)

    def fft_polynomial_multiplication(A, B):
        n = 1
        while n < len(A) + len(B):
            n <<= 1
        
        A_fft = np.fft.fft(A, n=n)
        B_fft = np.fft.fft(B, n=n)
        
        C_fft = A_fft * B_fft
        C = np.fft.ifft(C_fft).real
        
        return C[:len(A) + len(B) - 1]

使用 PyTorch 的 FFT 优化的多项式乘法 (FFT Optimized Polynomial Multiplication with PyTorch)

    def torch_fft_polynomial_multiplication(A, B):
        n = 1
        while n < len(A) + len(B):
            n <<= 1
        
        A_tensor = torch.tensor(A, dtype=torch.complex64)
        B_tensor = torch.tensor(B, dtype=torch.complex64)
        
        A_fft = torch.fft.fft(A_tensor, n=n)
        B_fft = torch.fft.fft(B_tensor, n=n)
        
        C_fft = A_fft * B_fft
        C = torch.fft.ifft(C_fft).real.numpy()
        
        return C[:len(A) + len(B) - 1]


Naive Polynomial Multiplication took 56457.79 milliseconds.

Result:
484.7532+1512.5681x-1948.8516x^2-2726.9115x^3-6721.8728x^4-2907.8580x^5-735.5199x^6+6265.3615x^7-2189.8554x^8-14600.6087x^9

FFT Optimized Polynomial Multiplication took 1.99 milliseconds.

Result:
484.7532+1512.5681x-1948.8516x^2-2726.9115x^3-6721.8728x^4-2907.8580x^5-735.5199x^6+6265.3615x^7-2189.8554x^8-14600.6087x^9

FFT Optimized Polynomial Multiplication with PyTorch took 12.97 milliseconds.

Result:
484.7656+1512.5312x-1948.8438x^2-2726.8828x^3-6721.9219x^4-2907.8672x^5-735.5469x^6+6265.3438x^7-2189.8125x^8-14600.6133x^9

可见并行化不完全是灵丹妙药，有时需要其他优化方法。

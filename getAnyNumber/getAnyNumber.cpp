//getAnyNumber.cpp : 拟合曲线
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
#include <cstdlib>

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
    bool first = true;
    for (int i = 0; i <= len; ++i) { // 从常数项开始，逐步打印到最高次项
        double coef = A[i];
        if (coef == 0) continue; // 跳过零项
        double abs_coef = std::abs(coef);

        if (!first) {
            if (coef > 0) std::cout << " + ";
            else std::cout << " - ";
        } else if (coef < 0) {
            std::cout << "-"; // 第一个非零项是负数时加负号
        }

        // 打印系数部分（注意：系数为 ±1 时特殊处理）
        if (abs_coef != 1 || i == 0) { // 常数项即使为1也要显示
            // std::cout<<i;
            std::cout << abs_coef;
        }


        // 打印变量部分
        if (i > 0) {
            std::cout << "x";
            if (i > 1) {
                std::cout << "^" << i;
            }
        }

        first = false;
    }

    if (first) {
        std::cout << "0"; // 所有系数都为0的情况
    }

    std::cout << std::endl;
}




// 计算多项式的系数
std::vector<double> CalculateCoefficients(const std::vector<double>& An, double target) {
    int n = An.size();


    // 初始化系数向量
    std::vector<double> ConsOri(n+1, 0);

    // 计算每个系数
    for (int i = 0; i < n+1; ++i) {
        double product = 1;
        for (int j = 0; j < n+1; ++j) {
            if (i != j) {
                product *= (j - i);
            }
        }
        if(i!=n)
        ConsOri[i] = An[i]/product;
        else ConsOri[i] = target/product;
    }

    // 展开多项式并化简
    std::vector<double> ans(n + 1, 0);

    for (int i = 0; i < n+1; ++i) {
        std::vector<double> tempAns(n + 1, 0); tempAns[0] = 1;
        for (int j = 0; j < n+1; ++j) {
            if (j == i)continue;
            else {
            std::vector<double>tempA(2, 1);
            tempA[0] = -j; tempA[1] = 1;

            std::vector<double>tempB(tempAns);
            openmpPolynomialMultiplication(tempA, tempB, tempAns);
            }

        }
        std::vector<double> tempCons(1, 0); tempCons[0]=ConsOri[i];
        std::vector<double>tempB(tempAns);
        openmpPolynomialMultiplication(tempCons, tempB, tempAns);
        for (int k = 0; k <= n; ++k) {
			ans[k] += tempAns[k];
		}

    }
    for (int k = 0; k <= n; ++k) {
        //因为double精度问题，一些值需要被视为0
        if (abs(ans[k]) < 1e-6) ans[k] = 0;

    }


    return ans;
}

int main(int argc, char* argv[]) {
    std::vector<double> An ; // 已知序列
    double target = 114514; // 目标值 t
    if (argc > 1) {
        // 如果有命令行参数
        for (int i = 1; i < argc - 1; ++i) {
            An.push_back(std::atof(argv[i])); // 将前 n-1 个参数作为 An
        }
        target = std::atof(argv[argc - 1]); // 最后一个参数为 target
    } else {
        // 没有参数时使用默认值
        An = {1, 5, 9, 15, 25};
    }
    //输出已知序列
    std::cout << "已知序列为: " << std::endl;
    for(int i = 0; i < An.size(); i++){
		std::cout << An[i] << " ";
	}
    std::cout << std::endl << "目标为: " <<target  <<std::endl;


    // 计算多项式的系数
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> ans = CalculateCoefficients(An, target);
    auto end = std::chrono::high_resolution_clock::now();

    // 输出多项式系数
    std::cout << "多项式系数为: " << std::endl;
    PrintV(ans, ans.size()); // 输出所有项

    // 输出计算时间
    std::cout << "计算时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}
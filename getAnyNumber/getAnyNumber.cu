//getAnyNumber.cu : 拟合曲线
// nvcc -o output/getAnyNumber_cuda getAnyNumber/getAnyNumber.cu -lcufft -Xcompiler -fopenmp -std=c++17
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
#include <cufft.h>

// CUDA 核函数定义
__global__ void multiplyComplex(cufftDoubleComplex* a, const cufftDoubleComplex* b, int n);
__global__ void normalizeArray(cufftDoubleComplex* arr, int n);

// GPU 多项式乘法接口
void GpuPolynomialMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& result);

// 拉格朗日插值主函数
std::vector<double> CudaCalculateCoefficients(const std::vector<double>& An, double target);

// 输出辅助函数
void PrintV(const std::vector<double>& A, int len);

__global__ void multiplyComplex(cufftDoubleComplex* a, const cufftDoubleComplex* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double real = a[i].x * b[i].x - a[i].y * b[i].y;
        double imag = a[i].x * b[i].y + a[i].y * b[i].x;
        a[i].x = real;
        a[i].y = imag;
    }
}

__global__ void normalizeArray(cufftDoubleComplex* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i].x /= n;
        arr[i].y /= n;
    }
}

void GpuPolynomialMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& result) {
    int n = 1;
    while (n < A.size() + B.size()) {
        n <<= 1;
    }

    cufftDoubleComplex *d_fa, *d_fb;
    cudaMalloc(&d_fa, n * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_fb, n * sizeof(cufftDoubleComplex));

    std::vector<cufftDoubleComplex> h_fa(n), h_fb(n);
    for (int i = 0; i < n; ++i) {
        h_fa[i].x = (i < A.size()) ? A[i] : 0;
        h_fa[i].y = 0;

        h_fb[i].x = (i < B.size()) ? B[i] : 0;
        h_fb[i].y = 0;
    }

    cudaMemcpy(d_fa, h_fa.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fb, h_fb.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);

    cufftExecZ2Z(plan, d_fa, d_fa, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_fb, d_fb, CUFFT_FORWARD);

    multiplyComplex<<<(n + 255) / 256, 256>>>(d_fa, d_fb, n);
    cufftExecZ2Z(plan, d_fa, d_fa, CUFFT_INVERSE);
    normalizeArray<<<(n + 255) / 256, 256>>>(d_fa, n);

    std::vector<cufftDoubleComplex> res(n);
    cudaMemcpy(res.data(), d_fa, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    result.resize(n);
    for (int i = 0; i < n; ++i)
        result[i] = res[i].x;

    cufftDestroy(plan);
    cudaFree(d_fa);
    cudaFree(d_fb);
}

std::vector<double> CudaCalculateCoefficients(const std::vector<double>& An, double target) {
    int n = An.size();
    std::vector<double> ConsOri(n + 1, 0);

    // 计算系数
    for (int i = 0; i <= n; ++i) {
        double product = 1;
        for (int j = 0; j <= n; ++j) {
            if (i != j)
                product *= (j - i);
        }
        ConsOri[i] = (i != n) ? An[i] / product : target / product;
    }

    std::vector<double> ans(n + 1, 0);

    #pragma omp parallel for
    for (int i = 0; i <= n; ++i) {
        std::vector<double> tempPoly(1, 1);  // 初始化为 1

        for (int j = 0; j <= n; ++j) {
            if (j == i) continue;
            std::vector<double> factor = { -static_cast<double>(j), 1.0 };
            std::vector<double> newTemp;
            GpuPolynomialMultiply(tempPoly, factor, newTemp);
            tempPoly = newTemp;
        }

        std::vector<double> term;
        GpuPolynomialMultiply(tempPoly, std::vector<double>(1, ConsOri[i]), term);

        #pragma omp critical
        {
            for (int k = 0; k <= n; ++k) {
                if (k < term.size())
                    ans[k] += term[k];
            }
        }
    }

    // 清理极小值
    for (int k = 0; k <= n; ++k) {
        if (std::abs(ans[k]) < 1e-6)
            ans[k] = 0;
    }

    return ans;
}

void PrintV(const std::vector<double>& A, int len) {
    bool first = true;
    for (int i = 0; i <= len; ++i) {
        double coef = A[i];
        if (coef == 0) continue;
        double abs_coef = std::abs(coef);

        if (!first) {
            if (coef > 0) std::cout << " + ";
            else std::cout << " - ";
        } else if (coef < 0) {
            std::cout << "-";
        }

        if (abs_coef != 1 || i == 0)
            std::cout << abs_coef;

        if (i > 0) {
            std::cout << "x";
            if (i > 1)
                std::cout << "^" << i;
        }

        first = false;
    }

    if (first) std::cout << "0";
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<double> An; // 已知序列
    double target = 114514; // 目标值 t

    if (argc > 1) {
        for (int i = 1; i < argc - 1; ++i)
            An.push_back(std::atof(argv[i]));
        target = std::atof(argv[argc - 1]);
    } else {
        An = {1, 5, 9, 15, 25};
    }

    std::cout << "已知序列为: ";
    for (double val : An) std::cout << val << " ";
    std::cout << "\n目标为: " << target << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> ans = CudaCalculateCoefficients(An, target);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "多项式系数为:\n";
    PrintV(ans, ans.size());

    std::cout << "计算时间: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    return 0;
}
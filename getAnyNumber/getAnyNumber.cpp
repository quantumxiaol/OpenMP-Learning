//getAnyNumber.cpp : �ҹ��ɻ���������ֵĶ���ʽ
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


// ���ٸ���Ҷ�任
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

// ����ʽ�˷�
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

// ��� OpenMP �� FFT �Ż��Ķ���ʽ�˷�
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

// ��ӡ����ʽ��ǰlen��
void PrintV(const std::vector<double>& A, int len) {
    // ��������ʽ��ǰlen��
    for (int i = 0; i < len && i < A.size(); ++i) {
        // ���ϵ����Ϊ0�����ӡ����
        if (A[i] != 0) {
            if (i > 0) std::cout << "+"; // ���˵�һ���⣬������ǰ�涼��'+'��
            if (A[i] != 1 || i == 0) std::cout << A[i]; // ���ϵ������1���ߵ�ǰ���ǳ�������ӡϵ��
            if (i == 1) std::cout << "x"; // ��ǰ����һ����
            else if (i > 1) std::cout << "x^" << i; // ��ǰ���Ǹ���һ�ε���
        }
    }
    std::cout << std::endl;
}




// �������ʽ��ϵ��
std::vector<double> CalculateCoefficients(const std::vector<double>& An, double target) {
    int n = An.size();


    // ��ʼ��ϵ������
    std::vector<double> ConsOri(n+1, 0);

    // ����ÿ��ϵ��
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

    // չ������ʽ������
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
        //��Ϊdouble�������⣬һЩֵ��Ҫ����Ϊ0
        if (abs(ans[k]) < 1e-6) ans[k] = 0;

    }


    return ans;
}

int main() {
    // ʾ������
    std::vector<double> An = { 1,5,9}; // ��֪����
    double target = 114514; // Ŀ��ֵ t

    //�����֪����
    std::cout << "��֪����Ϊ: " << std::endl;
    for(int i = 0; i < An.size(); i++){
		std::cout << An[i] << " ";
	}
    std::cout << std::endl << "Ŀ��Ϊ: " <<target  <<std::endl;


    // �������ʽ��ϵ��
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> ans = CalculateCoefficients(An, target);
    auto end = std::chrono::high_resolution_clock::now();

    // �������ʽϵ��
    std::cout << "����ʽϵ��Ϊ: " << std::endl;
    PrintV(ans, ans.size()); // ���������

    // �������ʱ��
    std::cout << "����ʱ��: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}
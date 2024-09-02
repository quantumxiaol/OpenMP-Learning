import numpy as np
import time
import torch
import torch.fft
def naive_polynomial_multiplication(A, B):
    n = len(A) + len(B) - 1
    C = np.zeros(n)
    for i in range(len(A)):
        for j in range(len(B)):
            C[i + j] += A[i] * B[j]
    return C

def fft_polynomial_multiplication(A, B):
    n = 1
    while n < len(A) + len(B):
        n <<= 1
    
    A_fft = np.fft.fft(A, n=n)
    B_fft = np.fft.fft(B, n=n)
    
    C_fft = A_fft * B_fft
    C = np.fft.ifft(C_fft).real
    
    return C[:len(A) + len(B) - 1]

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

def main():
    print("Torch version: ", torch.__version__)
    print("NumPy version: ", np.__version__)

    # 初始化随机数生成器
    np.random.seed(0)
    
    # 长度为10000的多项式
    length = 10000
    A = np.random.uniform(-100, 100, length)
    B = np.random.uniform(-100, 100, length)

    # 打印多项式
    def print_poly(C, n=10):
        for i in range(n):
            if C[i] != 0:
                sign = "+" if C[i] >= 0 else "-"
                if i == 0:
                    print(f"{C[i]:.4f}", end="")
                elif i == 1:
                    print(f"{sign}{abs(C[i]):.4f}x", end="")
                else:
                    print(f"{sign}{abs(C[i]):.4f}x^{i}", end="")
        print()

    # 比较四种算法的结果和运行时间
    methods = [
        (naive_polynomial_multiplication, "Naive Polynomial Multiplication"),
        (fft_polynomial_multiplication, "FFT Optimized Polynomial Multiplication"),
        (torch_fft_polynomial_multiplication, "FFT Optimized Polynomial Multiplication with PyTorch")
    ]
    
    for method, name in methods:
        start = time.time()* 1000
        C = method(A, B)
        end = time.time()* 1000
        print(f"{name} took {end - start:.2f} milliseconds.")
        print("Result:")
        print_poly(C, n=10)
        print()

if __name__ == "__main__":
    main()
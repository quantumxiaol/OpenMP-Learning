import time
import numpy as np
import torch

print("PyTorch version: ", torch.__version__)
print("NumPy version: ", np.__version__)
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# 随机生成两个矩阵
M, N, P = 1000, 1000, 1000
A = torch.randn(M, N, device=device)
B = torch.randn(N, P, device=device)

# 使用 PyTorch 进行矩阵乘法
start_time = time.time()
C = torch.matmul(A, B)
end_time = time.time()

# 记录运行时间
torch_time = (end_time - start_time) * 1e6  # 转换为微秒
print(f"PyTorch CUDA matrix multiplication took {torch_time:.2f} microseconds.")


def matrix_multiply_python(A, B):
    m, n = A.shape
    n, p = B.shape
    
    C = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

# A = np.random.rand(M, N)
# B = np.random.rand(N, P)
A = A.cpu().numpy()
B = B.cpu().numpy()


# 使用 NumPy 进行矩阵乘法
start_time = time.time()
C_numpy = np.dot(A, B)
end_time = time.time()

numpy_time = (end_time - start_time) * 1e6  # 转换为微秒
print(f"Numpy matrix multiplication took {numpy_time:.2f} microseconds.")


# 使用纯 Python 的 for 循环进行矩阵乘法
start_time = time.time()
C_python = matrix_multiply_python(A, B)
end_time = time.time()

python_time = (end_time - start_time) * 1e6  # 转换为微秒
print(f"Python matrix multiplication took {python_time:.2f} microseconds.")
import random
import time
import numpy as np
import torch
import platform

# 随机生成大量的二维坐标点，这些点分布在边长为 2 的正方形内（因为单位圆的直径为 2）。
# 对于每一个生成的点，计算它到原点的距离，如果距离小于等于 1，则认为该点位于单位圆内。
def estimate_pi(num_samples):
    num_inside = 0
    for _ in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if (x**2 + y**2) <= 1:
            num_inside += 1
    pi_estimate = 4 * num_inside / num_samples
    return pi_estimate


def estimate_pi_numpy(num_samples):
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    distance = x**2 + y**2
    num_inside = np.sum(distance <= 1)
    pi_estimate = 4 * num_inside / num_samples
    return pi_estimate

def monte_carlo_pi_vectorized(num_samples):
    random_points = np.random.uniform(-1, 1, size=(num_samples, 2))
    distances = np.linalg.norm(random_points, axis=1)
    inside = np.sum(distances <= 1)
    pi_estimate = 4 * float(inside) / float(num_samples)
    return pi_estimate

def get_device():
    # 检测系统信息
    os_name = platform.system()
    machine_type = platform.machine()

    if os_name == "Darwin" and machine_type.lower() in ["arm64", "aarch64"]:
        # 在 Apple Silicon (M1/M2/M3/M4) 上
        if torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders)")
            return torch.device("mps")
        else:
            print("MPS is not available, falling back to CPU.")
            return torch.device("cpu")
    elif torch.cuda.is_available():
        # 在支持 CUDA 的系统上
        print("Using CUDA")
        return torch.device("cuda")
    else:
        # 默认使用 CPU
        print("Using CPU")
        return torch.device("cpu")


def estimate_pi_pytorch_cuda(num_samples):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = get_device()
    x = torch.rand(num_samples, device=device) * 2 - 1
    y = torch.rand(num_samples, device=device) * 2 - 1
    distance = x ** 2 + y ** 2
    num_inside = torch.sum(distance <= 1).item()
    pi_estimate = 4 * num_inside / num_samples
    return pi_estimate

def estimate_pi_pytorch_cuda(num_samples, batch_size=100000000):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 获取设备
    device = get_device()
    total_inside = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        x = torch.rand(current_batch_size, device=device) * 2 - 1
        y = torch.rand(current_batch_size, device=device) * 2 - 1
        distance = x ** 2 + y ** 2
        inside = (distance <= 1).sum().item()
        total_inside += inside

    pi_estimate = 4 * total_inside / num_samples
    return pi_estimate

batch_size = 100000000
n=1000000000

print(f"Number of samples set to:{n}")
# 估计圆周率，使用 n 个样本
start_time = time.time()
# estimated_pi = estimate_pi_pytorch_cuda(n)
estimated_pi = estimate_pi_pytorch_cuda(n, batch_size)
end_time = time.time()
print(f"Estimated value of Pi(CUDA): {estimated_pi}")
print(f"Time taken: {(end_time - start_time):.4f} seconds.")

# 估计圆周率，使用 n 个样本
start_time = time.time()
estimated_pi = estimate_pi_numpy(n)
end_time = time.time()
print(f"Estimated value of Pi(NumPy): {estimated_pi}")
print(f"Time taken: {(end_time - start_time):.4f} seconds.")

# 估计圆周率，使用 n 个样本
# start_time = time.time()
# estimated_pi = monte_carlo_pi_vectorized(n)
# end_time = time.time()
# print(f"Estimated value of Pi(Vectorized): {estimated_pi}")
# print(f"Time taken: {(end_time - start_time):.4f} seconds.")


# 估计圆周率，使用 n 个样本
start_time = time.time()
estimated_pi = estimate_pi(n)
end_time = time.time()
print(f"Estimated value of Pi: {estimated_pi}")
print(f"Time taken: {(end_time - start_time):.4f} seconds.")
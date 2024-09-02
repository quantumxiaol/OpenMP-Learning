import random
import time
import numpy as np
import torch

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

def estimate_pi_pytorch_cuda(num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(num_samples, device=device) * 2 - 1
    y = torch.rand(num_samples, device=device) * 2 - 1
    distance = x ** 2 + y ** 2
    num_inside = torch.sum(distance <= 1).item()
    pi_estimate = 4 * num_inside / num_samples
    return pi_estimate

def estimate_pi_pytorch_cuda(num_samples, batch_size=100000000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
n=1000000
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
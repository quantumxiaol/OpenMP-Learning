import numpy as np
from numpy.polynomial import Polynomial
import time

def calculate_coefficients(an, target):
    # 计算每个系数
    n = len(an)
    cons_ori = []
    
    for i in range(n + 1):
        product = 1
        for j in range(n + 1):
            if i != j:
                product *= (j-i)
        if i == n:
            cons_ori.append(target/product)
        else:
            cons_ori.append(an[i]/product)
    
    # 展开多项式并化简
    ans = np.zeros(n + 1)
    
    for i in range(n + 1):
        poly = Polynomial([1])
        # 累乘 (x - j) 形式的多项式
        for j in range(n + 1):
            if i != j and j <= n:
                if j == i:
                    continue
                temp_poly = Polynomial([-j, 1])
                poly = poly * temp_poly
        
        # 将当前多项式乘以对应的系数
        scaled_poly = poly * cons_ori[i]
        ans += scaled_poly.coef
    
    # 因为 float 精度问题，一些值需要被视为 0
    ans[np.abs(ans) < 1e-6] = 0
    
    return ans

if __name__ == "__main__":
    # 示例数据
    an = [1, 5, 9]  # 已知序列
    target = 114514  # 目标值 t

    print("已知序列为:", an)
    print("目标为:", target)

    # 计算多项式的系数
    start_time = time.time()
    ans = calculate_coefficients(an, target)
    end_time = time.time()

    # 输出多项式系数
    # print("多项式系数为:")
    # print(ans)
    # 输出多项式:
    print("多项式为:")
    print(Polynomial(ans))
    
    # 输出计算时间
    print("计算时间:", (end_time - start_time) * 1000, "ms")
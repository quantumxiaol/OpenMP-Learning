# 获得任意数字

有这样找规律的数字

    1 3 5 __

答案是多少？

他可以是7，也可以是任意数t，满足这样的式子

    y=A(x-2)(x-3)(x-4)+B(x-1)(x-3)(x-4)+C(x-1)(x-2)(x-4)+D(x-1)(x-2)(x-3)

带入1 3 5 t(x=1,2,3,4)可有

    -6A=1
    2B=2
    -2C=3
    -6D=t
  
这样可以解出ABCD，带入合并可以得到y的表达式。

## C++

现在算法可以处理任意数字，它接受数组的前几位vector<double>An,double target,

处理得到ABCDE等，（把t带入计算），

他们是由就是n个数相乘(x-1)(x-2)……(x-n)后每个抽走对应的n的式子，ConsOri[i] = target/product;

构成n个系数ABCDE等。

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

多项式乘法采用FFT和OMP加速计算。这里可能存在数值精度的问题

不难注意到

    1 3 5 114514

因为他们满足

    y=-1+-38171x+57250.5x^2+-19083.5x^3

### 运行结果

1 3 5 7 9 11 13

y=-1+2x

![result1](https://github.com/quantumxiaol/OpenMP-Learning/blob/png/gAN1.png)

1 5 9 114514

-1+-38171x+57250.5x^2+-19083.5x^3

![result1]([https://github.com/quantumxiaol/OpenMP-Learning/blob/main/png/gAN5.png)

在geogebra中绘制图像

![result1](https://github.com/quantumxiaol/OpenMP-Learning/blob/png/gAN4.png)

1 3 5 7 9 11 114514

![result1](https://github.com/quantumxiaol/OpenMP-Learning/blob/png/gAN2.png)

![result1](https://github.com/quantumxiaol/OpenMP-Learning/blob/png/gAN3.png)
此时double丢失一些数据 

## python 
NumPy也可以直接计算多项式

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

已知序列为: [1, 5, 9]

目标为: 114514

多项式为:

      -1.0 - 38171.0 x**1 + 57250.5 x**2 - 19083.5 x**3
      
计算时间: 0.9982585906982422 ms

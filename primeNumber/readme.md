# 质数筛选
质数的定义：在大于 1 的自然数中，除了 1 和它本身以外不再有其他因数的自然数。

## 暴力循环
因此对于每个数 x，我们可以从小到大枚举 [2, x−1] 中的每个数 y，判断 y 是否为 x 的因数。这样判断一个数是否为质数的时间复杂度最差情况下会到 O(n)。整体的时间复杂度为O(n^2)。

    void naive_prime_sieve(int n, std::vector<int>& PrimeNumber) {
        for (int i = 2; i <= n; ++i) {
            bool is_prime = true;
            for (int j = 2; j < i; ++j) {
                if (i % j == 0) {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime) {
                PrimeNumber.push_back(i);
            }
        }
    }

## 暴力循环优化
实际上，如果 y 是 x 的因数，那么y/x也必然是 x 的因数，因此我们只要校验 y 或者 y/x 即可。而如果我们每次选择校验两者中的较小数，则不难发现较小数一定落在 [2, sqrt(n)] 的区间中，因此我们只需要枚举 [2, sqrt(n)] 中的所有数即可，这样单次检查的时间复杂度从 O(n) 降低至了 O( sqrt(n))。整体的时间复杂度为O(n^1.5)。
    
    for (int j = 2; j * j <= i; ++j) {}

## OMP优化循环
可以使用OMP优化外层循环，每个线程负责处理一部分数据。
每个线程有自己的局部 threadLocalPrimes 向量，并在临界区中合并结果。

    void omp_prime_sieve(int n, std::vector<int>& PrimeNumber) {
    #pragma omp parallel
        {
            std::vector<int> threadLocalPrimes;
    #pragma omp for
            for (int i = 2; i <= n; ++i) {
                bool is_prime = true;
                for (int j = 2; j * j <= i; ++j) {
                    if (i % j == 0) {
                        is_prime = false;
                        break;
                    }
                }
                if (is_prime) {
                    threadLocalPrimes.push_back(i);
                }
            }
    
    #pragma omp critical
            for (int prime : threadLocalPrimes) {
                PrimeNumber.push_back(prime);
            }
        }
    }

## 埃氏筛
该算法由希腊数学家厄拉多塞（Eratosthenes）提出，称为厄拉多塞筛法，简称埃氏筛。

如果 x 是质数，那么大于 x 的 x 的倍数一定不是质数，从最小的质数2开始，标记2的所有倍数为合数；然后找到下一个未被标记的数3，再次标记它的所有倍数为合数；重复此过程直到达到上限。

这种方法显然不会将质数标记成合数；另一方面，当从小到大遍历到数 x 时，倘若它是合数，则它一定是某个小于 x 的质数 y 的整数倍，故根据此方法的步骤，我们在遍历到 y 时，就一定会在此时将 x 标记为 isPrime[x]=0。因此，这种方法也不会将合数标记为质数。

时间复杂度：O(nloglogn)

    void eratosthenes_sieve(int n, std::vector<int>& PrimeNumber) {
        std::vector<bool> is_prime(n + 1, true);
        is_prime[0] = is_prime[1] = false;
        for (int i = 2; i * i <= n; ++i) {
            if (is_prime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    is_prime[j] = false;
                }
            }
        }
        for (int i = 2; i <= n; ++i) {
            if (is_prime[i]) {
                PrimeNumber.push_back(i);
            }
        }
    }

## 线性筛
线性筛法是一种优化过的埃氏筛法，它通过记录每个数的最小质因数来提高效率。

与埃氏筛不同的是，标记过程不再仅当 x 为质数时才进行，而是对每个整数 x 都进行。对于整数 x，不再标记其所有的倍数 x*x,x*(x+1),…，而是只标记质数集合中的数与 x 相乘的数，即 x*primes[i] 且在发现 x mod primes[i]==0 的时候结束当前标记。

时间复杂度：O(n)

    void linear_sieve(int n, std::vector<int>& PrimeNumber) {
        std::vector<int> min_prime(n + 1, 0);
        for (int i = 2; i <= n; ++i) {
            if (min_prime[i] == 0) {
                min_prime[i] = i;
                PrimeNumber.push_back(i);
            }
            for (int j = 0; j < PrimeNumber.size() && i * PrimeNumber[j] <= n; ++j) {
                min_prime[i * PrimeNumber[j]] = PrimeNumber[j];
                if (i % PrimeNumber[j] == 0) {
                    break;
                }
            }
        }
    }


## 运行结果

n=100000

Naive Prime Sieve: 74647ms

Optimized Naive Prime Sieve: 145ms

OpenMP Prime Sieve: 21ms

Eratosthenes Sieve: 6ms

Linear Sieve: 9ms

n=100000000

Optimized Naive Prime Sieve: 90596ms

OpenMP Prime Sieve: 14130ms

Eratosthenes Sieve: 461ms

Linear Sieve: 578ms

Prime Sieve: 2 3 5 7 11 13 17 19 23 29

31 37 41 43 47 53 59 61 67 71

73 79 83 89 97 101 103 107 109 113

127 131 137 139 149 151 157 163 167 173

179 181 191 193 197 199 211 223 227 229

233 239 241 251 257 263 269 271 277 281

283 293 307 311 313 317 331 337 347 349

353 359 367 373 379 383 389 397 401 409

419 421 431 433 439 443 449 457 461 463

467 479 487 491 499 503 509 521 523 541

## 问题分析
埃氏筛法通常使用一个布尔数组来标记质数，这种数组访问模式更容易利用现代计算机的缓存机制。线性筛法可能涉及更多的随机访问，导致缓存命中率较低，从而影响性能。

埃氏筛法通常按照顺序访问数组，这对于现代处理器的缓存友好。而线性筛法可能涉及更复杂的跳转，导致内存访问模式不连续，从而影响性能。

## 参考
leetcode 204. 计数质数 https://leetcode.cn/problems/count-primes/description/

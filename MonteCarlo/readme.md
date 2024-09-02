# 蒙特卡洛方法
# 计算圆周率

## 使用C++实现，使用OMP优化

### 原始版本

蒙特卡洛方法是一种基于随机抽样的数值计算方法。具体来说，我们可以通过以下步骤估计 π(PI) 的值：

生成随机点：在正方形 [-1, 1][-1, 1] 内生成随机点。
判断点的位置：如果点落在单位圆内（即x^2+y^2≤1），则计数器加 1。
计算圆周率的估计值：设总共有 N个点，其中M个点落在单位圆内，则的估计值为 PI≈4N/M。

    double monteCarloPi(long numSamples) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        long inside = 0;
    
        for (long i = 0; i < numSamples; ++i) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x * x + y * y <= 1.0) {
                ++inside;
            }
        }
    
        return 4.0 * inside / numSamples;
    }

### 并行版本

利用 OpenMP 来并行化计算过程，具体步骤如下：

使用 std::atomic<long> 来保证线程安全，初始化全局计数器。

每个线程处理一部分随机点。

每个线程有自己的局部计数器 localInside，用于统计落在单位圆内的点数。

在线程完成自己的部分后，进入临界区更新全局的原子计数器 totalInside。

计算PI的估计值：根据所有线程汇总的计数器值来估计PI的值。

    #pragma omp parallel
    {
        // 线程内的局部计数器
        long localInside = 0;
    
        // 每个线程处理的数据量
        long chunkSize = numSamples / omp_get_num_threads();
        long start = omp_get_thread_num() * chunkSize;
        long end = start + chunkSize;
    
        // 确保最后一个线程处理所有剩余的数据
        if (omp_get_thread_num() == omp_get_num_threads() - 1) {
            end = numSamples;
        }
    
        // 处理每个线程的数据
        for (long i = start; i < end; ++i) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x * x + y * y <= 1.0) {
                ++localInside;
            }
        }
    
        // 在退出临界区之前更新 totalInside
        #pragma omp critical
        {
            totalInside += localInside;
        }
    }

运行结果

Non-parallel Monte Carlo Pi: 3.14166

Time taken: 23.9205 seconds

Parallel Monte Carlo Pi: 3.14165

Time taken: 3.67023 seconds

可见并行确实可以优化效率

## 使用Python实现
也可以使用Python对比，PyTorch和NumPy也能做。

通过随机生成二维坐标点并检查这些点是否落在单位圆内来估计圆周率。

    def estimate_pi(num_samples):
        num_inside = 0
        for _ in range(num_samples):
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if (x**2 + y**2) <= 1:
                num_inside += 1
        pi_estimate = 4 * num_inside / num_samples
        return pi_estimate

使用了NumPy库来加速计算过程，与estimate_pi相比，这里使用了向量化运算来代替循环。也可以进一步使用NumPy的linalg模块来计算所有点到原点的距离。

      def estimate_pi_numpy(num_samples):
          x = np.random.uniform(-1, 1, num_samples)
          y = np.random.uniform(-1, 1, num_samples)
          distance = x**2 + y**2
          num_inside = np.sum(distance <= 1)
          pi_estimate = 4 * num_inside / num_samples
      return pi_estimate

使用PyTorch创建了两个张量x和y，它们在GPU上生成随机数，然后计算距离并统计落在圆内的点数。当数据量特别大时，采用Batch一次计算一组数据。当batch_size为100000000是，大概占用4G多显存。

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
        
运行结果

使用python迭代10^9

Estimated value of Pi(CUDA): 3.14175348

Time taken: 1.8672 seconds.

Estimated value of Pi(NumPy): 3.1416956

Time taken: 2.2403 seconds.

Estimated value of Pi: 3.1416648

Time taken: 82.5886 seconds.

## 结论

### 统计结果

| 迭代次数 |                   |PyTorch       | NumPy        | Python(Pure) | Parallel(OMP) | C++(Pure) |
|-----------|-------------|--------------|--------------|---------------|-----------------|------------|
| 1.0E+03  | 估计值        | 3.184             | 3.148        | 3.176            | 3.092                |3.096        |
|          | 耗时(s)       | 1.3098            | 0               | 0.001            | 0.0009611        |1.85E-05   |
| 1.0E+04  | 估计值        | 3.1228            | 3.1608        | 3.136            | 3.1212                |3.1432        |
|          | 耗时(s)       | 1.2327            | 0               | 0.007            | 0.0007504        |0.0001718   |
| 1.0E+05  | 估计值        | 3.13676           | 3.14772       | 3.14904          | 3.13972                |3.14084      |
|          | 耗时(s)       | 1.2378            | 0.003          | 0.0898           | 0.0007252        |0.0021416   |
| 1.0E+06  | 估计值        | 3.1396            | 3.13882       | 3.13934          | 3.13879                |3.14121      |
|          | 耗时(s)       | 1.2643            | 0.0219         | 0.8527           | 0.0028313        |0.0157529   |
| 1.0E+07  | 估计值        | 3.1414204         | 3.1416688     | 3.1420068        | 3.14129                |3.14247      |
|          | 耗时(s)       | 1.3617            | 0.2234         | 8.4199           | 0.0213917        |0.167197    |
| 1.0E+08  | 估计值        | 3.14152588        | 3.14170072    | 3.14165484       | 3.14168                |3.14186      |
|          | 耗时(s)       | 1.4052            | 2.1987         | 84.3979          | 0.312717        |1.87495     |
| 1.0E+09  | 估计值        | 3.141673972       | 3.14155072    | -                | 3.14165                |3.14162      |
|          | 耗时(s)       | 1.7673            | 113.3363       | -                | 2.59984            |16.886       |
| 1.0E+10  | 估计值        | 3.141611628       | -             | -        |3.14165          | 3.14166                |
|          | 耗时(s)       | 1.8482            | -             | -        |3.81769          | 23.7907            |
| 1.0E+11  | 估计值        | 3.141591531       | -             | -        |3.14164          | 3.14165                |
|          | 耗时(s)       | 5.3125            | -             | -        |3.16754          | 20.3928            |
| 1.0E+12  | 估计值        | 3.141584097       | -             | -                | -                    |-            |
|          | 耗时(s)       | 40.7725           | -             | -                | -                    |-            |
| 1.0E+12  | 估计值        | 3.14159474        | -             | -                | -                    |-            |
|          | 耗时(s)       | 391.1378          | -             | -                | -                    |-            |

### 数值结果

随着迭代次数的增加，所有方法估计的π值都逐渐趋近于真实值3.141592653589793...。这是因为随着采样点数量的增加，蒙特卡洛方法的估计结果会越来越接近真实的数学期望。

对于较大的迭代次数，例如10^12，PyTorch方法得到的估计值为3.141584097，而另一个估计值为3.14159474，这已经非常接近π的真实值。

### 计算时间
Python(Pure): 纯Python实现的计算时间随着迭代次数增加呈指数级增长，这是因为它没有利用任何向量化操作或并行处理技术。

NumPy: 使用NumPy之后，即使是大规模的迭代也能在较短的时间内完成。例如，在10^8次迭代时，耗时约为2.2秒。

PyTorch: PyTorch方法表现出了较好的性能，在高迭代次数下依然能保持较快的速度。在10^12次迭代时，耗时大约为40秒左右。

Parallel(OMP): 使用OpenMP（Open Multi-Processing）进行并行化处理后，可以看到即使是在较小的迭代次数下，也有明显的速度优势。

C++(Pure): 纯C++实现显示了非常高的性能，在10^9次迭代时，耗时仅为16.886秒。

### 并行处理的影响

使用OpenMP这样的并行处理技术可以显著提高计算效率，特别是在迭代次数较多的情况下。对于大规模数据集，使用并行处理技术是提高性能的有效手段。

### 不同语言/库的比较

NumPy和PyTorch这样的库提供了高性能的数组操作接口，能够在不牺牲太多易用性的情况下大幅提升性能。

C++作为编译型语言，在底层操作上具有天然的优势，特别是在涉及到大量数值计算时，性能通常优于解释型语言如Python。

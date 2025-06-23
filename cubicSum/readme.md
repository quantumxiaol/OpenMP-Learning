# 计算数组中所有元素的立方和

计算立方和只需要

    for (int i = 0; i < length; i++) {
          sum += array[i] * array[i] * array[i];
    }
    
适合用并行化提高效率。

下面是一些函数用途。

    cubicSum计算数组中所有元素的立方和

    cubicSum_MT使用 OpenMP 的并行循环来计算数组中所有元素的立方和

    cubicSum_MTv2使用 OpenMP 的并行区域来计算数组中所有元素的立方和

    #pragma omp parallel num_threads(std::thread::hardware_concurrency()): 指示 OpenMP 创建一组并行线程，线程数量等于硬件的最大并发能力。

    #pragma omp atomic: 指示 OpenMP 在原子操作下更新 sum 变量，以避免竞态条件。

性能测量: 使用 std::chrono 库来测量串行和并行版本的执行时间，并输出结果。

并行开销: 创建和销毁线程、数据分割、上下文切换等都会带来开销。

并行计算特性: 如果任务粒度过小，开销可能会超过并行带来的好处；如果硬件不足以支持所需的线程数量，性能提升可能不明显。

## 运行结果

在MacOS(M4)上的运行结果

    (使用Cmake 编译)
    Serial sum: 2.025e+33
    Parallel sum: 2.025e+33
    Parallel sum v2: 2.025e+34
    Time for serial: 876ms
    Time for parallel: 86ms
    Time for parallel v2: 691ms

    (使用clang++ 编译)
    Serial sum: 2.025e+33
    Parallel sum: 2.025e+33
    Parallel sum v2: 2.025e+33
    Time for serial: 845ms
    Time for parallel: 84ms
    Time for parallel v2: 79ms

    (使用clang++ 编译 -O2 -fopenmp )
    ./output/cubicSum                                                                                  
    Serial sum: 2.025e+33
    Parallel sum: 2.025e+33
    Parallel sum v2: 2.025e+33
    Time for serial: 646ms
    Time for parallel: 27ms
    Time for parallel v2: 27ms

    (使用Cmake 编译 -O2 -fopenmp )
    Serial sum: 2.025e+33
    Parallel sum: 2.025e+33
    Parallel sum v2: 2.025e+33
    Time for serial: 485ms
    Time for parallel: 29ms
    Time for parallel v2: 27ms

在Ubuntu22.04(Intel i7-10875)上的运行结果

    ./output/cubicSum
    Serial sum: 2.025e+33
    Parallel sum: 2.025e+33
    Parallel sum v2: 2.025e+33
    Time for serial: 842ms
    Time for parallel: 134ms
    Time for parallel v2: 142ms

在 cubicSum_MTv2 中手动划分线程任务，并通过 #pragma omp atomic 来更新全局变量 sum，这种方式效率低且容易引入浮点误差或竞争条件。
在Macos M4上，造成锁竞争。

修改set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp ${OpenMP_CXX_FLAGS}")后运行结果正常
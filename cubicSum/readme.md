#计算数组中所有元素的立方和

计算立方和只需要

    for (int i = 0; i < length; i++) {
          sum += array[i] * array[i] * array[i];
    }
    
适合用并行化提高效率。

cubicSum计算数组中所有元素的立方和

cubicSum_MT使用 OpenMP 的并行循环来计算数组中所有元素的立方和

cubicSum_MTv2使用 OpenMP 的并行区域来计算数组中所有元素的立方和

#pragma omp parallel num_threads(std::thread::hardware_concurrency()): 指示 OpenMP 创建一组并行线程，线程数量等于硬件的最大并发能力。

#pragma omp atomic: 指示 OpenMP 在原子操作下更新 sum 变量，以避免竞态条件。


性能测量: 使用 std::chrono 库来测量串行和并行版本的执行时间，并输出结果。

并行开销: 创建和销毁线程、数据分割、上下文切换等都会带来开销。

并行计算特性: 如果任务粒度过小，开销可能会超过并行带来的好处；如果硬件不足以支持所需的线程数量，性能提升可能不明显。

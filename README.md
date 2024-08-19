# OpenMP-Learning
# OpenMP入门教程学习笔记

使用OpenMP优化c++程序


##序言

最近需要处理比较大的数据，需要学习OpenMP来提高效率。

OpenMP（Open Multi-Processing）是一个支持多平台共享内存并行编程的应用程序接口（API），它可以在C、C++和Fortran语言中使用。通过使用OpenMP，开发者可以编写能够在多核心、多处理器计算机上高效运行的并行程序。

Visual Studio中在配置属性 - C/C++ - OpenMP支持 - 是
可以启用OpenMP，然后#include <omp.h>


OpenMP 使用特殊的编译指令来控制并行行为。常见的指令包括：

#pragma omp parallel: 开始一个新的并行区域。
#pragma omp parallel for: 并行执行循环。
#pragma omp single: 执行单个线程的任务。
#pragma omp critical: 保护临界区。
#pragma omp barrier: 所有线程到达屏障点后继续执行。
#pragma omp task: 创建一个新任务。

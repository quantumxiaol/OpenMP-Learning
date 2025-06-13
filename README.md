# OpenMP-Learning

OpenMP学习笔记

使用OpenMP优化C++程序

## 序言

最近需要处理比较大的数据，需要学习OpenMP来提高效率。

OpenMP（Open Multi-Processing）是一个支持多平台共享内存并行编程的应用程序接口（API），它可以在C、C++和Fortran语言中使用。通过使用OpenMP，开发者可以编写能够在多核心、多处理器计算机上高效运行的并行程序。

Windows下在Visual Studio中在配置属性 - C/C++ - OpenMP支持 - 是
可以启用OpenMP，然后#include <omp.h>

Ubuntu下默认的 gcc 和 g++ 编译器是支持 OpenMP 的。添加-fopenmp参数来启用OpenMP。

OpenMP 使用特殊的编译指令来控制并行行为。常见的指令包括：

    #pragma omp parallel: 开始一个新的并行区域。
    #pragma omp parallel for: 并行执行循环。
    #pragma omp single: 执行单个线程的任务。
    #pragma omp critical: 保护临界区。
    #pragma omp barrier: 所有线程到达屏障点后继续执行。
    #pragma omp task: 创建一个新任务。

## 目录

- [立方之和](cubicSum/readme.md)
- [归并排序](mergeSort/readme.md)
- [点云处理](rippleDetect/readme.md)  检测点云涟漪
- [边缘提取](edgeDetect/readme.md)  卷积核提取边缘
- [蒙特卡罗方法](MonteCarlo/readme.md)  估计圆周率
- [多项式乘法](polynomialMultiplication/readme.md)
- [矩阵乘法](matrixMultiplication/readme.md) 对比了C++的Eigen库和Python的NumPy库
- [数字找规律](getAnyNumber/readme.md)  这个数字一定是114514，或者是1919810
- [质数筛选](primeNumber/readme.md)  给出1到n中的所有质数
- [字符画](ASCIIArt/readme.md)  把图片/视频转为字符画

## 编译和运行

### Linux(Ubuntu)

命令行下运行`g++ -fopenmp -o output/main main.cpp`，将生成可执行文件`main`。

VSCode中可以编辑任务，在.vscode/tasks.json中添加以下内容：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build openmp app",
            "type": "shell",
            "command": "g++",
            "args": [
                "-fopenmp", // 启用OpenMP支持
                "-o", "${workspaceFolder}/output/MonteCarlo", // 输出可执行文件名
                "${workspaceFolder}/MonteCarlo/MonteCarlo.cpp" // 输入源文件
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"],
            "detail": "Task to compile a C++ application with OpenMP support."
        }
    ]
}
```

运行`code --file=tasks.json`，然后点击左下角的运行按钮，或者使用快捷键`Ctrl+Shift+B`。

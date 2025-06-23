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

## 环境配置

### Windows(X64)

### Linux(Ubuntu)(X64)

### macOS(Apple Silicon)

先配置Homebrew。

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

安装Cmake

    brew install cmake

在 macOS 上，OpenMP 的支持不像在 Linux 或 Windows 上那样直接可用，因为 macOS 的默认编译器（Apple Clang）并不默认包含对 OpenMP 的支持。
安装带有 OpenMP 支持的 LLVM：

    brew install llvm

在 CMake 中可以这样做：

    set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

安装PCL和opencv

    brew install opencv
    brew install pcl

通常会被安装到 /usr/local/Cellar/ 目录下。例如：

OpenCV 可能位于 /usr/local/Cellar/opencv/版本号/
PCL 可能位于 /usr/local/Cellar/pcl/版本号/

这些目录包含了库文件、头文件等资源。Homebrew 还会在 /usr/local/include 和 /usr/local/lib 中创建符号链接，指向上述路径中的头文件和库文件，以便于开发使用。

安装libomp

    brew install libomp

安装llvm

    brew install llvm

llvm提供了 C、C++ 和 Objective-C 的编译器。libomp则可以让C++程序使用OpenMP。


可以通过命令`brew --prefix opencv`查看opencv的安装路径。其他同理。

    brew info opencv
    brew info pcl

配置./.vscode/c_cpp_properties.json
```json
{
  "configurations": [
    {
      "name": "Mac (Apple Silicon)",
      "includePath": [
        "${workspaceFolder}/**",
        "/opt/homebrew/include", // 包含 Homebrew 安装的所有包的头文件
        "/opt/homebrew/opt/libomp/include", // libomp 头文件路径
        "/opt/homebrew/opt/llvm/include", // 如果用的是 Homebrew 安装的 LLVM 版本
        "/opt/homebrew/opt/opencv/include",
        "/opt/homebrew/include/eigen3",
        "/opt/homebrew/opt/pcl/include",
        "/opt/homebrew/include/opencv4",
        "/opt/homebrew/include/pcl-1.15"
      ],
      "defines": [],
      "macFrameworkPath": [
        "/System/Library/Frameworks",
        "/Library/Frameworks"
      ],
      "compilerPath": "/opt/homebrew/opt/llvm/bin/clang++", // 使用 Homebrew 安装的 clang++,/usr/bin/clang++是Apple自带的
      "cStandard": "c17",
      "cppStandard": "c++17",
      "intelliSenseMode": "clang-arm64",
      "compileCommands": "${workspaceFolder}/build/compile_commands.json"
    }
  ],
  "version": 4
}
```

配置./.vscode/settings.json
```json
{
  "cmake.configureSettings": {
    "CMAKE_C_COMPILER": "/opt/homebrew/opt/llvm/bin/clang",
    "CMAKE_CXX_COMPILER": "/opt/homebrew/opt/llvm/bin/clang++"
  }
}
```

## 编译和运行

### Linux(Ubuntu)

#### 无需OpenCV和PCL的(Windows、Ubuntu)

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

### MacOS(MacOS 的PCL、OpenCV、OpenMP都需要额外配置)

```CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(MyOpenCVAndPCLProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定 LLVM 编译器路径（用于支持 OpenMP）
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

find_package(JsonCpp REQUIRED)
include_directories(${JSONCPP_INCLUDE_DIRS})

# 查找 PCL
find_package(PCL REQUIRED COMPONENTS common io visualization)

# 查找 OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

# 可执行文件
add_executable(my_app cubicSum/cubicSum.cpp)

# 链接库
target_link_libraries(my_app
    PRIVATE
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        JsonCpp::JsonCpp
        OpenMP::OpenMP_CXX
)
```

VScode
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "CMake: Configure",
      "type": "shell",
      "command": "cmake",
      "args": [
        "-S", "${workspaceFolder}",
        "-B", "${workspaceFolder}/build",
        "-G", "Unix Makefiles"
      ],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "problemMatcher": ["$cmake"],
      "label": "CMake: Configure",
      "detail": "Configures the project using CMake into 'build' folder"
    },
    {
      "label": "CMake: Build",
      "type": "shell",
      "command": "cmake",
      "args": ["--build", "${workspaceFolder}/build", "--target", "my_app", "--", "-j4"],
      "group": {
        "kind": "build",
        "isDefault": false
      },
      "problemMatcher": ["$gcc"],
      "label": "CMake: Build",
      "detail": "Builds the project using CMake and make"
    },
    {
      "label": "CMake: Clean",
      "type": "shell",
      "command": "rm",
      "args": ["-rf", "${workspaceFolder}/build/*"],
      "group": "none",
      "label": "CMake: Clean",
      "detail": "Removes all files in the build directory"
    },
    {
      "label": "CMake: Run",
      "type": "shell",
      "command": "./build/my_app",
      "group": "none",
      "dependsOn": ["CMake: Build"],
      "label": "CMake: Run",
      "detail": "Builds and runs the executable"
    }
  ]
}
```

使用CMakeLists.txt编译运行

    cd /Volumes/ZHITAI2T/dev/OpenMP-Learning
    rm -rf build/
    mkdir build && cd build

    cmake \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..

    make -j$(sysctl -n hw.ncpu)

    ./my_app

更换项目是修改CMakeLists.txt的add_executable(my_app cubicSum/cubicSum.cpp)
// mergeSort.cpp : 归并排序。

#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>

std::mutex mtx;

// 归并两个已排序的子向量
void merge(std::vector<int>& vec, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // 创建临时向量
    std::vector<int> L(n1), R(n2);

    // 复制数据到临时向量
    for (int i = 0; i < n1; i++)
        L[i] = vec[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = vec[mid + 1 + j];

    // 合并临时向量回到原向量
    int i = 0; // 初始索引 of 第一个子向量
    int j = 0; // 初始索引 of 第二个子向量
    int k = left; // 初始索引 of 合并后的子向量

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        }
        else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    // 复制剩余元素
    while (i < n1) {
        vec[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        vec[k] = R[j];
        j++;
        k++;
    }
}

// 插入排序
void insertionSort(std::vector<int>& vec, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = vec[i];
        int j = i - 1;
        while (j >= left && vec[j] > key) {
            vec[j + 1] = vec[j];
            j--;
        }
        vec[j + 1] = key;
    }
}

// 归并排序函数（非多线程版本）
void mergeSort(std::vector<int>& vec, int left, int right) {
    if (left >= right) {
        return; // 递归终止条件
    }

    int mid = left + (right - left) / 2;

    // 递归排序左右子向量
    mergeSort(vec, left, mid);
    mergeSort(vec, mid + 1, right);

    // 合并两个子向量
    merge(vec, left, mid, right);
}

// 归并排序函数（非多线程版本,较小时使用插入排序）
void mergeSort_V1(std::vector<int>& vec, int left, int right) {
    if (left >= right) {
        return; // 递归终止条件
    }

    int mid = left + (right - left) / 2;

    // 对较小的子数组使用插入排序
    const int INSERTION_SORT_THRESHOLD = 16;
    if (right - left < INSERTION_SORT_THRESHOLD) {
        insertionSort(vec, left, right);
    }
    else {

        for (int i = 0; i < 2; ++i) {
            if (i == 0) {
                mergeSort(vec, left, mid);
            }
            else {
                mergeSort(vec, mid + 1, right);
            }
        }
    }

    // 合并两个子向量
    merge(vec, left, mid, right);
}


// 多线程版本的归并排序
void mergeSort_MT(std::vector<int>& vec, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    // 对较小的子数组使用插入排序
    const int INSERTION_SORT_THRESHOLD = 16;
    if (right - left < INSERTION_SORT_THRESHOLD) {
        insertionSort(vec, left, right);
    }
    else {
        // 使用并行 for 循环来并行执行左右子数组的排序任务
#pragma omp parallel for
        for (int i = 0; i < 2; ++i) {
            if (i == 0) {
                mergeSort_MT(vec, left, mid);
            }
            else {
                mergeSort_MT(vec, mid + 1, right);
            }
        }
    }

    merge(vec, left, mid, right);
}

// 用于初始化向量的多线程函数
void fillVectorWithRandoms(std::vector<int>& vec, int start, int end, std::default_random_engine& engine) {
    std::uniform_int_distribution<int> dist(0, 100000);

    for (int i = start; i <= end; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        vec[i] = dist(engine);
    }
}

int main() {
    const int length = 10000000; // 向量长度
    std::vector<int> array(length);

    // 使用本地时间作为种子初始化随机数生成器
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    const int numThreads = 4;
    int segmentLength = length / numThreads;
    std::thread threads[numThreads];

    // 为每个线程创建独立的随机数生成器
    std::default_random_engine engines[numThreads];

    // 为每个生成器设置不同的种子
    for (int i = 0; i < numThreads; ++i) {
        engines[i].seed(seed + i);
    }

    // 创建并启动线程
    for (int i = 0; i < numThreads; ++i) {
        int start = i * segmentLength;
        int end = (i == numThreads - 1) ? length - 1 : (i + 1) * segmentLength - 1;
        threads[i] = std::thread(fillVectorWithRandoms, std::ref(array), start, end, std::ref(engines[i]));
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }



    // 复制向量以进行多线程排序
    std::vector<int> array_v1(array);
    std::vector<int> array_mt(array);
    std::vector<int> array_lib(array);

    // 计算非多线程排序的时间
    auto start1 = std::chrono::high_resolution_clock::now();
    mergeSort(array, 0, length - 1);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto start1_v1 = std::chrono::high_resolution_clock::now();
    mergeSort_V1(array_v1, 0, length - 1);
    auto end1_v1 = std::chrono::high_resolution_clock::now();

    // 计算多线程排序的时间
    auto start2 = std::chrono::high_resolution_clock::now();
    mergeSort_MT(array_mt, 0, length - 1);
    auto end2 = std::chrono::high_resolution_clock::now();

    // 使用标准库的排序函数
    auto start3 = std::chrono::high_resolution_clock::now();
    std::sort(array_lib.begin(), array_lib.end());
    auto end3 = std::chrono::high_resolution_clock::now();

    // 输出结果和耗时
    std::cout << "First 200 elements after sorting: ";
    for (int i = 0; i < 200; i++) {
        std::cout << array[i] << " ";
        if(i% 25 == 0) std::cout << std::endl;
    }
    std::cout << "..." << std::endl;

    std::cout << "First 200 elements after multi-threaded sorting: ";
    for (int i = 0; i < 200; i++) {
        std::cout << array_mt[i] << " ";
        if (i % 25 == 0) std::cout << std::endl;
    }
    std::cout << "..." << std::endl;

    std::cout<< "First 100 elements after std::sort: ";
    for (int i = 0; i < 200; i++) {
		std::cout << array_lib[i] << " ";
		if (i % 25 == 0) std::cout << std::endl;
	}
    std::cout << "..." << std::endl;

    std::cout << "Time for non-parallel merge sort: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;
    std::cout<< "Time for non-parallel merge sort with insertion sort: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1_v1 - start1_v1).count() << "ms" << std::endl;
    std::cout << "Time for parallel merge sort: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;
    std::cout<< "Time for std::sort: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << "ms" << std::endl;

    return 0;
}
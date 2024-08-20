// mergeSort.cpp : �鲢����

#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>

std::mutex mtx;

// �鲢�����������������
void merge(std::vector<int>& vec, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // ������ʱ����
    std::vector<int> L(n1), R(n2);

    // �������ݵ���ʱ����
    for (int i = 0; i < n1; i++)
        L[i] = vec[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = vec[mid + 1 + j];

    // �ϲ���ʱ�����ص�ԭ����
    int i = 0; // ��ʼ���� of ��һ��������
    int j = 0; // ��ʼ���� of �ڶ���������
    int k = left; // ��ʼ���� of �ϲ����������

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

    // ����ʣ��Ԫ��
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

// ��������
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

// �鲢���������Ƕ��̰߳汾��
void mergeSort(std::vector<int>& vec, int left, int right) {
    if (left >= right) {
        return; // �ݹ���ֹ����
    }

    int mid = left + (right - left) / 2;

    // �ݹ���������������
    mergeSort(vec, left, mid);
    mergeSort(vec, mid + 1, right);

    // �ϲ�����������
    merge(vec, left, mid, right);
}

// �鲢���������Ƕ��̰߳汾,��Сʱʹ�ò�������
void mergeSort_V1(std::vector<int>& vec, int left, int right) {
    if (left >= right) {
        return; // �ݹ���ֹ����
    }

    int mid = left + (right - left) / 2;

    // �Խ�С��������ʹ�ò�������
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

    // �ϲ�����������
    merge(vec, left, mid, right);
}


// ���̰߳汾�Ĺ鲢����
void mergeSort_MT(std::vector<int>& vec, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    // �Խ�С��������ʹ�ò�������
    const int INSERTION_SORT_THRESHOLD = 16;
    if (right - left < INSERTION_SORT_THRESHOLD) {
        insertionSort(vec, left, right);
    }
    else {
        // ʹ�ò��� for ѭ��������ִ���������������������
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

// ���ڳ�ʼ�������Ķ��̺߳���
void fillVectorWithRandoms(std::vector<int>& vec, int start, int end, std::default_random_engine& engine) {
    std::uniform_int_distribution<int> dist(0, 100000);

    for (int i = start; i <= end; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        vec[i] = dist(engine);
    }
}

int main() {
    const int length = 10000000; // ��������
    std::vector<int> array(length);

    // ʹ�ñ���ʱ����Ϊ���ӳ�ʼ�������������
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    const int numThreads = 4;
    int segmentLength = length / numThreads;
    std::thread threads[numThreads];

    // Ϊÿ���̴߳��������������������
    std::default_random_engine engines[numThreads];

    // Ϊÿ�����������ò�ͬ������
    for (int i = 0; i < numThreads; ++i) {
        engines[i].seed(seed + i);
    }

    // �����������߳�
    for (int i = 0; i < numThreads; ++i) {
        int start = i * segmentLength;
        int end = (i == numThreads - 1) ? length - 1 : (i + 1) * segmentLength - 1;
        threads[i] = std::thread(fillVectorWithRandoms, std::ref(array), start, end, std::ref(engines[i]));
    }

    // �ȴ������߳����
    for (auto& thread : threads) {
        thread.join();
    }



    // ���������Խ��ж��߳�����
    std::vector<int> array_v1(array);
    std::vector<int> array_mt(array);
    std::vector<int> array_lib(array);

    // ����Ƕ��߳������ʱ��
    auto start1 = std::chrono::high_resolution_clock::now();
    mergeSort(array, 0, length - 1);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto start1_v1 = std::chrono::high_resolution_clock::now();
    mergeSort_V1(array_v1, 0, length - 1);
    auto end1_v1 = std::chrono::high_resolution_clock::now();

    // ������߳������ʱ��
    auto start2 = std::chrono::high_resolution_clock::now();
    mergeSort_MT(array_mt, 0, length - 1);
    auto end2 = std::chrono::high_resolution_clock::now();

    // ʹ�ñ�׼���������
    auto start3 = std::chrono::high_resolution_clock::now();
    std::sort(array_lib.begin(), array_lib.end());
    auto end3 = std::chrono::high_resolution_clock::now();

    // �������ͺ�ʱ
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
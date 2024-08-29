// conv.cpp : ���������
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>
#include <chrono>
#include <thread>

// �����
std::vector<std::vector<float>> kernelXV= { {1, 1, 1}, {0, 0, 0}, {-1, -1, -1} };
std::vector<std::vector<float>> kernelYV= { {1, 0, -1}, {1, 0, -1}, {1, 0, -1} };


// ��ͨ���������
void convolve(const cv::Mat& src, const std::vector<std::vector<float>>& kernel, cv::Mat& dst) {
    int padding = kernel.size() / 2;
    cv::Mat paddedSrc;
    cv::copyMakeBorder(src, paddedSrc, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    dst.create(src.size(), CV_32F); // ���������;���
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            float val = 0.0f;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    val += paddedSrc.at<uchar>(y + ky, x + kx) * kernel[ky][kx];
                }
            }
            dst.at<float>(y, x) = val;
        }
    }
}

// ��Ե��⺯��
void edgeDetection(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat gradX, gradY;
    convolve(src, kernelXV, gradX);
    convolve(src, kernelYV, gradY);

    cv::Mat mag;
    gradX.convertTo(gradX, CV_32F);
    gradY.convertTo(gradY, CV_32F);
    cv::sqrt(gradX.mul(gradX) + gradY.mul(gradY), mag);
    dst.create(src.size(), CV_8U);

    // ��magת��Ϊuchar����
    mag.convertTo(dst, CV_8U); // �����ת���� uchar ����
}

// ��ͨ���������
void convolveOMP(const cv::Mat& src, const std::vector<std::vector<float>>& kernel, cv::Mat& dst) {
    int padding = kernel.size() / 2;
    cv::Mat paddedSrc;
    cv::copyMakeBorder(src, paddedSrc, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    dst.create(src.size(), CV_32F); // ���������;���

    // ���л�����,�ڲ��л���ʼʱ�ͽ��������ȵط���������߳�
#pragma omp parallel for schedule(static)
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            float val = 0.0f;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    val += paddedSrc.at<uchar>(y + ky, x + kx) * kernel[ky][kx];
                }
            }
            dst.at<float>(y, x) = val;
        }
    }
}

// OpenMP���л���Ե��⺯��
void edgeDetectionOMP(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat gradX, gradY;
    convolveOMP(src, kernelXV, gradX);
    convolveOMP(src, kernelYV, gradY);
    //cv::imshow("Origin", gradX);
    cv::Mat mag;
    gradX.convertTo(gradX, CV_32F); //cv::imshow("Origin", gradX);
    gradY.convertTo(gradY, CV_32F);
    cv::sqrt(gradX.mul(gradX) + gradY.mul(gradY), mag);
    //cv::imshow("Origin", mag);
    mag.convertTo(dst, CV_8U); // �����ת���� uchar ����

}


int main() {
    std::string path = "C:\\work\\testData\\Ayabe.png";

    cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (!src.data) {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }
    
    // ���ô��ڴ�С
    int windowWidth = src.cols;
    int windowHeight = src.rows;

    cv::namedWindow("Origin Image Admrie Vega", cv::WINDOW_NORMAL);
    cv::resizeWindow("Origin Image Admrie Vega", windowWidth, windowHeight);

    cv::namedWindow("Standard Edge Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Standard Edge Detection", windowWidth, windowHeight);

    cv::namedWindow("OpenMP Edge Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("OpenMP Edge Detection", windowWidth, windowHeight);


    cv::Mat dst, dst_omp;
    auto start = std::chrono::high_resolution_clock::now();
    edgeDetection(src, dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Standard edge detection took " << duration << " microseconds." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    edgeDetectionOMP(src, dst_omp);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "OpenMP edge detection took " << duration << " microseconds." << std::endl;
    cv::imshow("Origin Image Admrie Vega", src);
    cv::imshow("Standard Edge Detection", dst);
    cv::imshow("OpenMP Edge Detection", dst_omp);
    cv::waitKey(0);

    return 0;
}
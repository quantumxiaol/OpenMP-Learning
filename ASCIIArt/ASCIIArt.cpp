//ASCIIArt.cpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <string>
#include <chrono>
// 定义字符集，用于表示不同的灰度级别
const char* ascii_chars = " .,-~:;=!*#$@";

// 函数：将灰度值映射到字符
char grayToChar(int gray) {
    // 确保灰度值在0-255之间
    if (gray < 0) gray = 0;
    if (gray > 255) gray = 255;
    // 映射灰度到字符
    return ascii_chars[gray * (strlen(ascii_chars) - 1) / 255];
}

void ReadImg(const std::string& imgPath) {
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return ;
    }

    // 缩放图片
    cv::resize(img, img, cv::Size(), 0.2, 0.2, cv::INTER_AREA);

    // 遍历所有像素
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // 获取当前像素的灰度值
            uchar gray = img.at<uchar>(i, j);
            // 转换灰度值为字符
            std::cout << grayToChar(gray);
        }
        std::cout << std::endl;
    }
}

// 函数：读取并显示视频
void ReadVideo(const std::string& videoPath) {
    cv::VideoCapture cap(videoPath); // 打开视频文件
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return;
    }

    cv::Mat frame, grayFrame;

    while (true) {
        cap >> frame; // 获取下一帧
        if (frame.empty()) break; // 如果没有获取到帧，则退出循环

        // 转换为灰度图
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // 缩放图片以适应终端窗口
        cv::resize(grayFrame, grayFrame, cv::Size(), 0.2, 0.2, cv::INTER_AREA);

        // 清除之前的帧
        system("cls"); // Windows系统使用 cls 命令清屏
        // 在Unix/Linux系统上可以使用 system("clear");

        // 遍历所有像素并打印ASCII字符
        for (int i = 0; i < grayFrame.rows; ++i) {
            for (int j = 0; j < grayFrame.cols; ++j) {
                uchar gray = grayFrame.at<uchar>(i, j);
                std::cout << grayToChar(gray);
            }
            std::cout << std::endl;
        }

        // 控制帧率，这里设置为每秒30帧
        cv::waitKey(33); // 33毫秒大约是30帧/秒
    }
}

// 函数：将ASCII字符渲染到图像上
cv::Mat renderAsciiArt(const cv::Mat& grayFrame, int blockWidth, int blockHeight) {
    int frameWidth = grayFrame.cols;
    int frameHeight = grayFrame.rows;

    // 计算输出图像尺寸
    int outputWidth = (frameWidth / blockWidth) * 8;  // 每个字符用8像素宽
    int outputHeight = (frameHeight / blockHeight) * 16; // 每个字符用16像素高

    // 创建一个空白图像
    cv::Mat outputImage(outputHeight, outputWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < frameHeight; y += blockHeight) {
        for (int x = 0; x < frameWidth; x += blockWidth) {
            // 确保ROI不超出图像边界
            int width = std::min(blockWidth, frameWidth - x);
            int height = std::min(blockHeight, frameHeight - y);

            // 获取当前块
            cv::Rect roi(x, y, width, height);
            cv::Mat block = grayFrame(roi);

            // 计算当前块的平均灰度值
            cv::Scalar mean = cv::mean(block);
            int avgGray = static_cast<int>(mean[0]);

            // 将灰度值映射到字符
            char ch = grayToChar(avgGray);

            // 计算当前字符的位置
            int outputX = (x / blockWidth) * 8;
            int outputY = (y / blockHeight) * 16;

            // 绘制字符
            cv::putText(outputImage, std::string(1, ch), cv::Point(outputX, outputY + 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, 8);
        }
    }

    return outputImage;
}


// 函数：处理视频并生成字符画视频（使用OpenMP优化）
// 函数：处理视频并生成字符画视频
void SaveVideoOmp(const std::string& inputPath, const std::string& outputPath) {
    cv::VideoCapture cap(inputPath); // 打开输入视频文件
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file: " << inputPath << std::endl;
        return;
    }

    // 获取视频属性
    double fps = cap.get(CV_CAP_PROP_FPS);
    int frameWidth = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    // 定义块大小
    int blockWidth = 8;  // 块宽度
    int blockHeight = 16; // 块高度

    // 计算输出视频的尺寸
    int outputWidth = (frameWidth / blockWidth) * 8;  // 每个字符用8像素宽
    int outputHeight = (frameHeight / blockHeight) * 16; // 每个字符用16像素高

    // 打印输入路径以供调试
    std::cout << "Input video path: " << inputPath << std::endl;
    std::cout << "Output video path: " << outputPath << std::endl;
    std::cout << "Input video properties:" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Frame Width: " << frameWidth << std::endl;
    std::cout << "Frame Height: " << frameHeight << std::endl;
    std::cout << "Output video dimensions: " << outputWidth << "x" << outputHeight << std::endl;

    // 尝试使用不同的编解码器
    //std::vector<int> fourccs = { CV_FOURCC('X', '2', '6', '4'), CV_FOURCC('X', 'V', 'I', 'D'), CV_FOURCC('D', 'I', 'V', 'X') };
    std::vector<int> fourccs = { CV_FOURCC('X', 'V', 'I', 'D'), CV_FOURCC('D', 'I', 'V', 'X') };

    cv::VideoWriter writer;
    for (const auto& fourcc : fourccs) {
        writer.open(outputPath, fourcc, fps, cv::Size(outputWidth, outputHeight), true);
        if (writer.isOpened()) {
            break;
        }
    }

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open the output video file: " << outputPath << std::endl;
        return;
    }

    cv::Mat frame, grayFrame, asciiFrame;

    while (true) {
        cap >> frame; // 获取下一帧
        if (frame.empty()) break; // 如果没有获取到帧，则退出循环

        // 转换为灰度图
        cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);

        // 渲染ASCII艺术到图像
        asciiFrame = renderAsciiArt(grayFrame, blockWidth, blockHeight);

        // 写入帧到输出视频
        writer.write(asciiFrame);
    }

    // 释放资源
    cap.release();
    writer.release();
}


int main(int argc, char** argv) {
    //std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    //std::cout << "Build information: " << cv::getBuildInformation() << std::endl;
    std::string inpath = "2.mp4";
    std::string outpath = "02.avi";

    auto start = std::chrono::steady_clock::now();
    SaveVideoOmp(inpath, outpath);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


    return 0;
}
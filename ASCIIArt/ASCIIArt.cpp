//ASCIIArt.cpp
//
//
// MacOS
// /opt/homebrew/opt/llvm/bin/clang++ \
  -std=c++17 -fopenmp -O2 \
  ASCIIArt/ASCIIArt.cpp -o output/ASCIIArt \
  $(pkg-config --cflags --libs opencv4 pcl_common pcl_io pcl_kdtree pcl_search)
// run with ./output/ASCIIArt 8 16 in.mp4 out.avi
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <string>
#include <chrono>
#include <filesystem> 
#include <cstdlib>     // for std::exit
#include <stdexcept>   // for std::invalid_argument, std::out_of_range

// 定义字符集，用于表示不同的灰度级别
const char* ascii_chars = " .,-~:;=!*#$@";

namespace fs = std::filesystem;

std::string resolvePath(const std::string& path) {
    fs::path filePath(path);
    if (filePath.is_absolute()) {
        return path;
    } else {
        // 如果是相对路径，则相对于 ./TestData/ 目录
        fs::path testDataDir("./TestData");
        return (testDataDir / filePath).make_preferred().string();
    }
}

void clearScreen() {
    std::cout << "\033[2J\033[H"; // ANSI escape sequence to clear screen and move cursor to home position
    std::cout.flush();
}
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
// 使用OpenMP加速ASCII渲染
void ReadVideoOmp(
    const std::string& videoPath,
    int blockWidth = 8,   // 每个字符代表的像素宽度
    int blockHeight = 16 // 每个字符代表的像素高度
) {
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

        // 缩放图像以适配终端大小（可选）
        // double scale = 0.4; // 可调整缩放比例
        // cv::resize(grayFrame, grayFrame, cv::Size(), scale, scale, cv::INTER_AREA);

        int rows = grayFrame.rows;
        int cols = grayFrame.cols;

        int numBlocksX = cols / blockWidth;
        int numBlocksY = rows / blockHeight;

        // 创建二维字符数组用于存储ASCII字符
        char** asciiGrid = new char*[numBlocksY];
        for (int i = 0; i < numBlocksY; ++i) {
            asciiGrid[i] = new char[numBlocksX];
        }

        // 并行化处理每个块
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < numBlocksY; ++y) {
            for (int x = 0; x < numBlocksX; ++x) {
                int startY = y * blockHeight;
                int startX = x * blockWidth;

                long sum = 0;
                int count = 0;

                for (int j = 0; j < blockHeight && (startY + j) < rows; ++j) {
                    for (int i = 0; i < blockWidth && (startX + i) < cols; ++i) {
                        sum += grayFrame.at<uchar>(startY + j, startX + i);
                        ++count;
                    }
                }

                int avgGray = static_cast<int>(sum / count);
                asciiGrid[y][x] = grayToChar(avgGray);
            }
        }

        // 清屏（Windows用cls，Mac/Linux用clear）
        // system("clear");
        clearScreen();

        // 输出ASCII艺术
        for (int y = 0; y < numBlocksY; ++y) {
            for (int x = 0; x < numBlocksX; ++x) {
                std::cout << asciiGrid[y][x];
            }
            std::cout << std::endl;
        }

        // 释放内存
        for (int i = 0; i < numBlocksY; ++i) {
            delete[] asciiGrid[i];
        }
        delete[] asciiGrid;

        // 控制帧率（约30fps）
        cv::waitKey(33); // 约30ms
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
void SaveVideoOmp(
    const std::string& inputPath, 
    const std::string& outputPath,
    int blockWidth = 8,   // 每个字符代表的像素宽度
    int blockHeight = 16 // 每个字符代表的像素高度
) {
    cv::VideoCapture cap(inputPath); // 打开输入视频文件
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file: " << inputPath << std::endl;
        return;
    }

    // 获取视频属性
    // opencv 2 CV_CAP_PROP_FPS-> opencv4 cv::CAP_PROP_FPS
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 定义块大小
    // int blockWidth = 8;  // 块宽度
    // int blockHeight = 16; // 块高度

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
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        // opencv 2 CV_BGR2GRAY->cv::COLOR_BGR2GRAY

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
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <blockWidth><blockWidth><input_path> [output_path]" << std::endl;
        return -1;
    }
    int blockWidth = 8;   // 每个字符代表的像素宽度
    int blockHeight = 16; // 每个字符代表的像素高度
        try {
        blockWidth = std::stoi(argv[1]);
        blockHeight = std::stoi(argv[2]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Block size must be an integer." << std::endl;
        return -1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: Block size out of range." << std::endl;
        return -1;
    }

    // 检查 block 尺寸是否合法
    if (blockWidth <= 0 || blockHeight <= 0) {
        std::cerr << "Error: Block width and height must be positive integers." << std::endl;
        return -1;
    }
    std::string inpath = resolvePath(argv[3]);
    std::string outpath;

    // 确保输入路径存在
    if (!fs::exists(inpath)) {
        std::cerr << "Input file does not exist: " << inpath << std::endl;
        return -1;
    }

    if (argc == 5) {
        outpath = resolvePath(argv[4]);

        // 输出路径处理，确保输出目录存在
        fs::path outputPath(outpath);
        fs::create_directories(outputPath.parent_path());

        std::cout << "Using input path: " << inpath << std::endl;
        std::cout << "Using output path: " << outpath << std::endl;

        auto start = std::chrono::steady_clock::now();
        SaveVideoOmp(inpath, outpath,blockWidth,blockHeight);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    } else {
        std::cout << "Using input path: " << inpath << std::endl;
        std::cout << "No output path specified. Playing video in terminal." << std::endl;

        auto start = std::chrono::steady_clock::now();
        ReadVideoOmp(inpath,blockWidth,blockHeight);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }

    return 0;
}
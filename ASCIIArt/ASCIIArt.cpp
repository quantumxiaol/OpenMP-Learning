//ASCIIArt.cpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <string>
#include <chrono>
// �����ַ��������ڱ�ʾ��ͬ�ĻҶȼ���
const char* ascii_chars = " .,-~:;=!*#$@";

// ���������Ҷ�ֵӳ�䵽�ַ�
char grayToChar(int gray) {
    // ȷ���Ҷ�ֵ��0-255֮��
    if (gray < 0) gray = 0;
    if (gray > 255) gray = 255;
    // ӳ��Ҷȵ��ַ�
    return ascii_chars[gray * (strlen(ascii_chars) - 1) / 255];
}

void ReadImg(const std::string& imgPath) {
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return ;
    }

    // ����ͼƬ
    cv::resize(img, img, cv::Size(), 0.2, 0.2, cv::INTER_AREA);

    // ������������
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // ��ȡ��ǰ���صĻҶ�ֵ
            uchar gray = img.at<uchar>(i, j);
            // ת���Ҷ�ֵΪ�ַ�
            std::cout << grayToChar(gray);
        }
        std::cout << std::endl;
    }
}

// ��������ȡ����ʾ��Ƶ
void ReadVideo(const std::string& videoPath) {
    cv::VideoCapture cap(videoPath); // ����Ƶ�ļ�
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return;
    }

    cv::Mat frame, grayFrame;

    while (true) {
        cap >> frame; // ��ȡ��һ֡
        if (frame.empty()) break; // ���û�л�ȡ��֡�����˳�ѭ��

        // ת��Ϊ�Ҷ�ͼ
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // ����ͼƬ����Ӧ�ն˴���
        cv::resize(grayFrame, grayFrame, cv::Size(), 0.2, 0.2, cv::INTER_AREA);

        // ���֮ǰ��֡
        system("cls"); // Windowsϵͳʹ�� cls ��������
        // ��Unix/Linuxϵͳ�Ͽ���ʹ�� system("clear");

        // �����������ز���ӡASCII�ַ�
        for (int i = 0; i < grayFrame.rows; ++i) {
            for (int j = 0; j < grayFrame.cols; ++j) {
                uchar gray = grayFrame.at<uchar>(i, j);
                std::cout << grayToChar(gray);
            }
            std::cout << std::endl;
        }

        // ����֡�ʣ���������Ϊÿ��30֡
        cv::waitKey(33); // 33�����Լ��30֡/��
    }
}

// ��������ASCII�ַ���Ⱦ��ͼ����
cv::Mat renderAsciiArt(const cv::Mat& grayFrame, int blockWidth, int blockHeight) {
    int frameWidth = grayFrame.cols;
    int frameHeight = grayFrame.rows;

    // �������ͼ��ߴ�
    int outputWidth = (frameWidth / blockWidth) * 8;  // ÿ���ַ���8���ؿ�
    int outputHeight = (frameHeight / blockHeight) * 16; // ÿ���ַ���16���ظ�

    // ����һ���հ�ͼ��
    cv::Mat outputImage(outputHeight, outputWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < frameHeight; y += blockHeight) {
        for (int x = 0; x < frameWidth; x += blockWidth) {
            // ȷ��ROI������ͼ��߽�
            int width = std::min(blockWidth, frameWidth - x);
            int height = std::min(blockHeight, frameHeight - y);

            // ��ȡ��ǰ��
            cv::Rect roi(x, y, width, height);
            cv::Mat block = grayFrame(roi);

            // ���㵱ǰ���ƽ���Ҷ�ֵ
            cv::Scalar mean = cv::mean(block);
            int avgGray = static_cast<int>(mean[0]);

            // ���Ҷ�ֵӳ�䵽�ַ�
            char ch = grayToChar(avgGray);

            // ���㵱ǰ�ַ���λ��
            int outputX = (x / blockWidth) * 8;
            int outputY = (y / blockHeight) * 16;

            // �����ַ�
            cv::putText(outputImage, std::string(1, ch), cv::Point(outputX, outputY + 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, 8);
        }
    }

    return outputImage;
}


// ������������Ƶ�������ַ�����Ƶ��ʹ��OpenMP�Ż���
// ������������Ƶ�������ַ�����Ƶ
void SaveVideoOmp(const std::string& inputPath, const std::string& outputPath) {
    cv::VideoCapture cap(inputPath); // ��������Ƶ�ļ�
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file: " << inputPath << std::endl;
        return;
    }

    // ��ȡ��Ƶ����
    double fps = cap.get(CV_CAP_PROP_FPS);
    int frameWidth = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    // ������С
    int blockWidth = 8;  // ����
    int blockHeight = 16; // ��߶�

    // ���������Ƶ�ĳߴ�
    int outputWidth = (frameWidth / blockWidth) * 8;  // ÿ���ַ���8���ؿ�
    int outputHeight = (frameHeight / blockHeight) * 16; // ÿ���ַ���16���ظ�

    // ��ӡ����·���Թ�����
    std::cout << "Input video path: " << inputPath << std::endl;
    std::cout << "Output video path: " << outputPath << std::endl;
    std::cout << "Input video properties:" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Frame Width: " << frameWidth << std::endl;
    std::cout << "Frame Height: " << frameHeight << std::endl;
    std::cout << "Output video dimensions: " << outputWidth << "x" << outputHeight << std::endl;

    // ����ʹ�ò�ͬ�ı������
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
        cap >> frame; // ��ȡ��һ֡
        if (frame.empty()) break; // ���û�л�ȡ��֡�����˳�ѭ��

        // ת��Ϊ�Ҷ�ͼ
        cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);

        // ��ȾASCII������ͼ��
        asciiFrame = renderAsciiArt(grayFrame, blockWidth, blockHeight);

        // д��֡�������Ƶ
        writer.write(asciiFrame);
    }

    // �ͷ���Դ
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
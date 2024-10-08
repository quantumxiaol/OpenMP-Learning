# 字符画
# ASCIIArt

用不同的字符代表不同灰度的像素
  
    const char* ascii_chars = " .,-~:;=!*#$@";

将灰度值映射到字符

    char grayToChar(int gray) {
        // 确保灰度值在0-255之间
        if (gray < 0) gray = 0;
        if (gray > 255) gray = 255;
        // 映射灰度到字符
        return ascii_chars[gray * (strlen(ascii_chars) - 1) / 255];
    }

可以逐个像素的读出来转为对应字符。不过有个问题是像素和字符的大小不一样，像素是正方形的，而字符则是个窄一点的矩形，一种解决方法是用一个窗口处理一片像素，而不是单个的。

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

如果要处理视频也是类似的思路。

在空白的一帧上根据窗口绘制字符。

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


打开一个本地视频，读取每一帧，转换，写入视频。我的opencv2.4.13读取不了H.265的视频。写入以.avi的格式保存。

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

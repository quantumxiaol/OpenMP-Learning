# 图像卷积运算边缘提取

可以使用卷积核来提取图像边缘。

定义两个3x3的卷积核 kernelX 和 kernelY，分别用于计算水平和垂直方向上的梯度。

std::vector<std::vector<float>> kernelXV= { {1, 1, 1}, {0, 0, 0}, {-1, -1, -1} };

std::vector<std::vector<float>> kernelYV= { {1, 0, -1}, {1, 0, -1}, {1, 0, -1} };

一个提取水平方向的边缘，一个提取竖直方向的边缘。在边缘处，梯度变化是最大的。

    void convolve(const cv::Mat& src, const std::vector<std::vector<float>>& kernel, cv::Mat& dst) {
        int padding = kernel.size() / 2;
        cv::Mat paddedSrc;
        cv::copyMakeBorder(src, paddedSrc, padding, padding, padding, padding, cv::BORDER_REPLICATE);
    
        dst.create(src.size(), CV_32F); // 创建浮点型矩阵
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

卷积过程是可以并行化提高效率的，这就是CUDA比CPU更适合做这个的原因。这里可以用OMP优化这个循环。

在for循环前添加#pragma omp parallel for schedule(static)即可。在并行化开始时就将迭代均匀地分配给各个线程。

使用两个卷积核处理后得到上下边缘和左右边缘，再叠加即可。

检测结果

![ayabe](https://github.com/quantumxiaol/OpenMP-Learning/blob/main/png/%E6%A3%80%E6%B5%8B%E7%BB%93%E6%9E%9C.png)

数据显示开OMP要节省近一半的时间。

为这碟醋包了这顿饺子。（指阿雅贝）

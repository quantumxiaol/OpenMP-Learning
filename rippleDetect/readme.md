# 点云涟漪检测

下面介绍一个检测点云涟漪的算法。基于C++的PCL。

注意虽然测试数据为生成简单采用了同一高度，但实际数据比较复杂，可能存在多个不同高度的平面及涟漪，不能简单的标红z以上的值。

## 涟漪生成 (cloudRipGen)

该函数用于生成包含随机涟漪的点云数据。

初始化点云: 定义点云的宽度、高度和点之间的步长，然后创建一个空的点云对象。

生成平面点云: 在给定的宽度和高度范围内，以指定的步长生成一系列点，初始高度设为0。

生成涟漪: 使用随机数生成器选取涟漪中心的位置、涟漪的半径和强度。对于每一个涟漪，遍历所有点云中的点，如果点到涟漪中心的距离小于涟漪的半径，则根据距离和强度更新该点的高度。

保存点云: 将生成的点云保存为 .pcd 文件。
  
    int cloudRipGen()
    {
    
    }


## 异常点检测 (PointLocalNExistNDetection)

此函数检测点云中是否存在局部区域内的点高度差异超过一定阈值的情况，并将满足条件的点保存到一个新的点云中。

加载点云: 从文件中加载点云数据。

构建KD树: 使用PCL库中的KD树结构来加速邻域搜索。

异常点检测: 对每个点进行半径搜索，查找其周围半径内的其他点。如果找到的点中高度差异超过阈值的点的数量大于等于 n，则认为该点是异常点，并将其添加到输出点云中。

保存异常点: 将检测出的异常点保存到一个新的 .pcd 文件中。

    //使用kdtree搜索方法检测局部存在n个高度差超过阈值的点
    void PointLocalNExistNDetection(
        std::string path,
        std::string inputfilename,
        std::string outputfilename,
        float radius,
        float threshold,
        int n
    ) {
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    
        std::stringstream ss;
        ss << path << inputfilename;
        if (pcl::io::loadPCDFile<PointT>(ss.str(), *cloud) == -1)
        {
            PCL_ERROR("Error while reading the point cloud file!\n");
            return;
        }
    
        std::cout << "Loaded " << cloud->points.size() << " data points from " << inputfilename << std::endl;
    
        // Create a KDTree for the search method of the extraction.
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        tree->setInputCloud(cloud);
    
        for (size_t i = 0; i < cloud->points.size(); ++i)
        {
            std::vector<int> indices;
            std::vector<float> square_dists;
            tree->radiusSearch(i, radius, indices, square_dists);
    
            int outliers_count = 0;
            for (auto index : indices)
            {
                double diff = std::abs(cloud->points[i].z - cloud->points[index].z);
                if (diff > threshold)
                {
                    outliers_count++;
                    if (outliers_count >= n)
                        break;
                }
            }
    
            if (outliers_count >= n)
            {
                cloud_filtered->points.push_back(cloud->points[i]);
            }
        }
    
        std::cout << "Detected " << cloud_filtered->points.size() << " outlier points." << std::endl;
        cloud_filtered->width = cloud_filtered->points.size();
        cloud_filtered->height = 1;
    
        std::stringstream output_stream;
        output_stream << path << outputfilename;
        if (pcl::io::savePCDFileASCII(output_stream.str(), *cloud_filtered) == -1)
        {
            PCL_ERROR("Error while saving the point cloud file!\n");
            return;
        }
    
        std::cout << "Saved " << cloud_filtered->points.size() << " data points to " << outputfilename << std::endl;
    }


## 采用多线程优化

将检测的点分配给多个线程，每个线程检测一段。

    #pragma omp parallel num_threads(threadnum)
        {
            size_t thread_id = omp_get_thread_num();
            size_t num_threads = omp_get_num_threads();
    
            // Calculate the start and end indices for this thread
            size_t start = (cloud->points.size() / num_threads) * thread_id;
            size_t end = (thread_id != num_threads - 1) ?
                (cloud->points.size() / num_threads) * (thread_id + 1) :
                cloud->points.size();
    
            for (size_t i = start; i < end; ++i)
            {
                //半径搜索
            }
        }
    
        // Merge local clouds into the main cloud_filtered
        for (const auto& local_cloud : local_clouds)
        {
            cloud_filtered->points.insert(cloud_filtered->points.end(), local_cloud.begin(), local_cloud.end());
        }

运行结果

Loaded 9000000 data points from ripple.pcd

Detected 136995 outlier points.

Saved 136995 data points to result-ripple1.pcd

Time taken: 79.4942 s

Loaded 9000000 data points from ripple.pcd

Detected 136995 outlier points.

Saved 136995 data points to result-ripple2.pcd

Time taken: 47.3285 s

Loaded 9000000 data points from ripple.pcd

Downsampled to 2257738 data points.

Detected 37150 outlier points.

Saved 37150 data points to result-ripple3.pcd

Time taken: 38.0032 s

可以看出使用多线程可以提高效率。

![涟漪侧视图](https://github.com/quantumxiaol/OpenMP-Learning/blob/main/png/%E6%B6%9F%E6%BC%AA1.png)

![涟漪](https://github.com/quantumxiaol/OpenMP-Learning/blob/main/png/%E6%B6%9F%E6%BC%AA2.png)

![涟漪检测结果](https://github.com/quantumxiaol/OpenMP-Learning/blob/main/png/%E6%B6%9F%E6%BC%AA3.png)

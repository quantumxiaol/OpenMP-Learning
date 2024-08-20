#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>
#include <thread>
typedef pcl::PointXYZ PointT;
int cloudRipGen()
{
    // 设置点云参数
    int width = 3000; // 平面宽度
    int height = 3000; // 平面高度
    float step = 1.0f; // 点之间的步长
    int num_ripples = 250; // 涟漪数量
    float max_ripple_radius = 25.0f; // 最大涟漪影响半径
    float max_ripple_strength = 13.0f; // 最大涟漪强度

    // 创建点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 生成平面点云
    for (int i = 0; i < width; i += step)
    {
        for (int j = 0; j < height; j += step)
        {
            pcl::PointXYZ point;
            point.x = i;
            point.y = j;
            point.z = 0.0f; // 初始z值设为0

            cloud->push_back(point);
        }
    }

    // 随机选择多个点作为涟漪中心
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, width);
    std::uniform_real_distribution<> radius_dis(0.0f, max_ripple_radius);
    std::uniform_real_distribution<> strength_dis(0.0f, max_ripple_strength);

    // 生成涟漪
    for (int i = 0; i < num_ripples; ++i)
    {
        float center_x = dis(gen);
        float center_y = dis(gen);
        float ripple_radius = radius_dis(gen);
        float ripple_strength = strength_dis(gen);

        // 应用涟漪效果
        for (auto& point : cloud->points)
        {
            float dx = point.x - center_x;
            float dy = point.y - center_y;
            float distance = sqrt(dx * dx + dy * dy);

            if (distance <= ripple_radius)
            {
                // 计算Z值的变化
                point.z += ripple_strength * (1 - distance / ripple_radius);
            }
        }
    }

    // 输出点云
    pcl::io::savePCDFileASCII("C:\\work\\OpenMPLearning\\ripple.pcd", *cloud);

    return 0;
}


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


//使用kdtree搜索方法检测局部存在n个高度差超过阈值的点，多线程版本
void PointLocalNExistNDetectionMT(
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

    // Get number of hardware threads
    int threadnum = std::thread::hardware_concurrency();

    // Allocate local point clouds for each thread
    std::vector<std::vector<PointT>> local_clouds(threadnum);

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
                local_clouds[thread_id].push_back(cloud->points[i]);
            }
        }
    }

    // Merge local clouds into the main cloud_filtered
    for (const auto& local_cloud : local_clouds)
    {
        cloud_filtered->points.insert(cloud_filtered->points.end(), local_cloud.begin(), local_cloud.end());
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

//使用kdtree搜索方法检测局部存在n个高度差超过阈值的点，多线程版本，输入滤波
void PointLocalNExistNDetectionMTv1(
    std::string path,
    std::string inputfilename,
    std::string outputfilename,
    float radius,
    float threshold,
    int n,
    float voxel_size
) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>());

    std::stringstream ss;
    ss << path << inputfilename;
    if (pcl::io::loadPCDFile<PointT>(ss.str(), *cloud) == -1)
    {
        PCL_ERROR("Error while reading the point cloud file!\n");
        return;
    }

    std::cout << "Loaded " << cloud->points.size() << " data points from " << inputfilename << std::endl;

    // Create a VoxelGrid filter for down-sampling
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(*cloud_downsampled);

    std::cout << "Downsampled to " << cloud_downsampled->points.size() << " data points." << std::endl;

    // Create a KDTree for the search method of the extraction.
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_downsampled);

    // Get number of hardware threads
    int threadnum = std::thread::hardware_concurrency();

    // Allocate local point clouds for each thread
    std::vector<std::vector<PointT>> local_clouds(threadnum);

#pragma omp parallel num_threads(threadnum)
    {
        size_t thread_id = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();

        // Calculate the start and end indices for this thread
        size_t start = (cloud_downsampled->points.size() / num_threads) * thread_id;
        size_t end = (thread_id != num_threads - 1) ?
            (cloud_downsampled->points.size() / num_threads) * (thread_id + 1) :
            cloud_downsampled->points.size();

        for (size_t i = start; i < end; ++i)
        {
            std::vector<int> indices;
            std::vector<float> square_dists;
            tree->radiusSearch(i, radius, indices, square_dists);

            int outliers_count = 0;
            for (auto index : indices)
            {
                double diff = std::abs(cloud_downsampled->points[i].z - cloud_downsampled->points[index].z);
                if (diff > threshold)
                {
                    outliers_count++;
                    if (outliers_count >= n)
                        break;
                }
            }

            if (outliers_count >= n)
            {
                local_clouds[thread_id].push_back(cloud_downsampled->points[i]);
            }
        }
    }

    // Merge local clouds into the main cloud_filtered
    for (const auto& local_cloud : local_clouds)
    {
        cloud_filtered->points.insert(cloud_filtered->points.end(), local_cloud.begin(), local_cloud.end());
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

int main(){
    std::string path = "C:\\work\\OpenMPLearning\\";
    std::string inputfilename = "ripple.pcd";
    std::string outputfilename = "result-ripple.pcd";

    std::chrono::duration<double> elapsed;
    auto end1 = std::chrono::high_resolution_clock::now();
    PointLocalNExistNDetection(path, inputfilename, "result-ripple1.pcd", 5, 1.0, 3);
    auto end2 = std::chrono::high_resolution_clock::now();
    elapsed = end2 - end1;
    std::cout << "Time taken: " << elapsed.count() << " s" << std::endl;

    auto end3 = std::chrono::high_resolution_clock::now();
    PointLocalNExistNDetectionMT(path, inputfilename, "result-ripple2.pcd", 5, 1.0, 3);
    auto end4 = std::chrono::high_resolution_clock::now();
    elapsed = end4 - end3;
    std::cout << "Time taken: " << elapsed.count() << " s" << std::endl;

    auto end5 = std::chrono::high_resolution_clock::now();
    PointLocalNExistNDetectionMTv1(path, inputfilename, "result-ripple3.pcd", 5, 1.0, 3,2.0);
    auto end6 = std::chrono::high_resolution_clock::now();
    elapsed = end6 - end5;
    std::cout << "Time taken: " << elapsed.count() << " s" << std::endl;

    return 0;

}
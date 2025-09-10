/**
 * @author Seongchang Park (scsc1125@gmail.com)
 * @date 2022-04-14 20:04
 */
#include <lidar_lane_detection/lane_detection.h>

LaneDetector::LaneDetector()
{
  bool init_result = init();
  ROS_ASSERT(init_result);

  theta_r = 180 * M_PI/ 180;
}
LaneDetector::~LaneDetector()
{
}

// Init function
bool LaneDetector::init()
{
  os1_sub_ = nh_.subscribe("/input", 10, &LaneDetector::OS1PointCloudCallback, this);
  origin_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/origin_output", 10);
  lane_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/lane_output", 10);

  lane_udp_pub_ = nh_.advertise<lidar_msgs::Lane>("/lane_pcl_parsed", 10);

  lane_line_left_pub_  = nh_.advertise<visualization_msgs::Marker>("/lane_line_left", 1);
  lane_line_right_pub_ = nh_.advertise<visualization_msgs::Marker>("/lane_line_right", 1);

  return true;
}

// Callback 함수 for OS1 point cloud msgs
void LaneDetector::OS1PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msgs)
{
  // Convert point cloud to PCL native point cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);

  pcl::fromROSMsg(*cloud_msgs, *cloud_ptr);

  // for(int i = 0; i < cloud_ptr->size(); i++)
  // {
  //   cloud_ptr->points[i].z += OS1_HEIGHT;
  // }
  //pcl::copyPointCloud(*cloud_ptr, *cloud_filtered_ptr);

  constexpr float kFilterResolution = 0.2;    // 0.2 -> voxel leaf size
  //                               x    y     z
  const Eigen::Vector4f kMinPoint( 0,  -6,  -1, 1);    // -50 -6 -3 1
  const Eigen::Vector4f kMaxPoint( 60,  6, 0.1, 1);    // 60 6.5 4 1
  auto filter_cloud = FilterCloud(cloud_ptr, kFilterResolution, kMinPoint, kMaxPoint);

  constexpr int kMaxIterations = 1000;         // 1000
  constexpr float kDistanceThreshold = 0.3;
  auto segment_cloud = SegmentPlane(filter_cloud, kMaxIterations, kDistanceThreshold);
                                  // filter_cloud
                                  
  pcl::PassThrough<pcl::PointXYZI> pass;
  pass.setInputCloud(segment_cloud.second);
  pass.setFilterFieldName("intensity");
  pass.setFilterLimits(70, 130);
  // pass.setFilterLimits(125, 150);
  pass.setFilterLimitsNegative(false);
  pass.filter(*cloud_filtered_ptr);
  // pcl::copyPointCloud(*segment_cloud.second, *cloud_filtered_ptr);

  cout << "lane size: " << cloud_filtered_ptr->size() << endl;
  auto lane_msgs = lidar_msgs::Lane();

  for (int i = 0; i < 200; i++)
  {
    if (i < cloud_filtered_ptr->size())
    {
      lane_msgs.points[i].x = cloud_filtered_ptr->points[i].x;
      lane_msgs.points[i].y = cloud_filtered_ptr->points[i].y;
      lane_msgs.points[i].z = cloud_filtered_ptr->points[i].z;
      lane_msgs.points[i].intensity = cloud_filtered_ptr->points[i].intensity;
    }
    else
    {
      lane_msgs.points[i].x = 0;
      lane_msgs.points[i].y = 0;
      lane_msgs.points[i].z = 0;
      lane_msgs.points[i].intensity = -1;
    }
  }

  lane_udp_pub_.publish(lane_msgs);

  // 원래 데이터 publish위해서
  sensor_msgs::PointCloud2 origin_point_cloud;
  pcl::toROSMsg(*cloud_ptr, origin_point_cloud);

  pcl::toROSMsg(*cloud_filtered_ptr, ros_output_);
  ros_output_.header.frame_id = "os_sensor";

  lane_pub_.publish(ros_output_);
  origin_pub_.publish(origin_point_cloud);

  pcl::PointCloud<pcl::PointXYZI>::Ptr left_lane(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr right_lane(new pcl::PointCloud<pcl::PointXYZI>());

  const float y_gap  = 1.2f;
  const float y_band = 0.8f;
  splitCloudLeftRight(cloud_filtered_ptr, left_lane, right_lane, y_gap, y_band);

  // 2) x 기준 정렬 & 샘플 간격 간단히 정리
  sortByXAndThin(left_lane, 0.1f);
  sortByXAndThin(right_lane, 0.1f);

  // 3) 마커 생성 (색상/두께는 취향대로)
  auto left_marker = createLaneLineMarker(left_lane, "os_sensor", "lane", 0, 0.0f, 1.0f, 0.0f, 0.07);
  auto right_marker = createLaneLineMarker(right_lane, "os_sensor", "lane", 1, 1.0f, 0.8f, 0.0f, 0.07);

  // 포인트가 2개 이상일 때만 퍼블리시(한 점이면 선이 안 보임)
  if (left_marker.points.size() >= 2)  lane_line_left_pub_.publish(left_marker);
  if (right_marker.points.size() >= 2) lane_line_right_pub_.publish(right_marker);
}

visualization_msgs::Marker LaneDetector::createLaneLineMarker(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const std::string& frame_id,
    const std::string& ns,
    int id,
    float r, float g, float b,
    double width)
{
  visualization_msgs::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = ros::Time::now();
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::Marker::LINE_STRIP;
  m.action = visualization_msgs::Marker::ADD;

  m.scale.x = width;     // 선 두께 (m)
  m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;

  m.pose.orientation.w = 1.0; // 기본

  m.points.reserve(cloud->size());
  for (const auto& pt : cloud->points) {
    geometry_msgs::Point p;
    p.x = pt.x; p.y = pt.y; p.z = pt.z;
    m.points.push_back(p);
  }

  // 라인이 안 보이지 않게 적어도 2포인트 이상일 때만 쓰는 게 좋아요.
  return m;
}

void LaneDetector::splitCloudLeftRight(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& left_out,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& right_out,
    float y_gap, float y_band)
{
  left_out->clear();
  right_out->clear();
  left_out->reserve(in->size());
  right_out->reserve(in->size());

  for (const auto& p : in->points) {
    // 좌:  y_gap < y <= y_gap + y_band
    if ( (p.y > y_gap) && (p.y <= y_gap+y_band) ) {
      left_out->push_back(p);
    }
    // 우: -y_gap - y_band <= y < -y_gap
    else if ( (p.y >= -y_gap-y_band) && (p.y <  -y_gap) ) {
      right_out->push_back(p);
    }
  }
}

void LaneDetector::sortByXAndThin(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    float min_dx)
{
  if (cloud->empty()) return;

  std::sort(cloud->points.begin(), cloud->points.end(),
            [](const pcl::PointXYZI& a, const pcl::PointXYZI& b){ return a.x < b.x; });

  pcl::PointCloud<pcl::PointXYZI> thinned;
  thinned.points.reserve(cloud->points.size());

  float last_x = std::numeric_limits<float>::lowest();
  for (const auto& p : cloud->points) {
    if (thinned.points.empty() || (p.x - last_x) >= min_dx) {
      thinned.points.push_back(p);
      last_x = p.x;
    }
  }
  *cloud = thinned;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr LaneDetector::FilterCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        float filterRes, const Eigen::Vector4f& minPoint, const Eigen::Vector4f& maxPoint)
{
  /*** Voxel grid point reduction and region based filtering ***/
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());

  // Filter point cloud that is out of region of interest
  pcl::CropBox<pcl::PointXYZI> region(true);
  region.setMin(minPoint);
  region.setMax(maxPoint);
  region.setInputCloud(cloud);        // filtered_cloud
  region.filter(*filtered_cloud);     // filtered_cloud

  std::vector<int> indices;
  
  // Filter point cloud on the roof of host vehicle
  region.setMin(Eigen::Vector4f(-2, -0.9, -1.9, 1));  // -1.5 1.7 -1 1
  region.setMax(Eigen::Vector4f(2.2, 0.9, 0, 1));  // 2.6 1.7 -0.4 1
  region.setInputCloud(filtered_cloud);   // filtered_cloud
  region.filter(indices);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  for (int index : indices)
  {
      inliers->indices.push_back(index);
  }

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZI> extract;
  // Extract the point cloud on roof
  extract.setInputCloud(filtered_cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);  // true
  extract.filter(*filtered_cloud);

  return filtered_cloud;
}

std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr,  pcl::PointCloud<pcl::PointXYZI>::Ptr> LaneDetector::SeparateClouds(
        const pcl::PointIndices::Ptr& inliers, const  pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
  // Create two new point clouds, one cloud with obstacles and other with segmented plane
     pcl::PointCloud<pcl::PointXYZI>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZI>());
     pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZI>());


    // Copy inliers point cloud as plane
    for (int index : inliers->indices)
    {
        plane_cloud->points.push_back(cloud->points[index]);
    }

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    // Extract the inliers so that we can get obstacles point cloud
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);

    std::pair< pcl::PointCloud<pcl::PointXYZI>::Ptr,  pcl::PointCloud<pcl::PointXYZI>::Ptr> segResult(obstacle_cloud, plane_cloud);
    return segResult;
}

// Use the pair object to hold your segmented results for the obstacle point cloud and the road point cloud
std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> LaneDetector::SegmentPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    //auto startTime = std::chrono::steady_clock::now();

    /*** PCL IMPLEMENTATION START ***/
	// Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    // Segment the largest planar component from the input cloud
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cerr << "Could not estimate a planar model for the given dataset" << std::endl;
    }

    /*auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;*/

    std::pair<typename pcl::PointCloud<pcl::PointXYZI>::Ptr, typename pcl::PointCloud<pcl::PointXYZI>::Ptr> segResult = SeparateClouds(inliers, cloud);
    return segResult;
}
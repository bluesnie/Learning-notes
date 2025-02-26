###### datetime:2025/02/25 11:15

###### author:nzb

# [感知示例](https://moveit.picknik.ai/main/doc/examples/perception_pipeline/perception_pipeline_tutorial.html)

## 介绍

`MoveIt` 允许通过 `Octomap` 无缝集成 `3D` 传感器。

## 入门

即使您从未完成入门教程，您仍然可以运行本教程的演示。但建议从入门教程的步骤开始，以便更好地理解本教程中的内容。在本节中，我们将逐步介绍如何配置机器人上的 `3D` 传感器与 `MoveIt` 集成。`MoveIt` 中处理 `3D` 感知的主要组件是 **Occupancy Map Updater**（占用地图更新器）。更新器使用插件架构来处理不同类型的输入。目前 `MoveIt` 中可用的插件包括：

- **PointCloud Occupancy Map Updater**：可以处理点云数据（`sensor_msgs/msg/PointCloud2`）。
- **Depth Image Occupancy Map Updater**：可将深度图像作为输入信息（`sensor_msgs/msg/Image`）。

## 如何创建用于 Octomap 生成的 3D 点云数据

在本教程中，您可以使用 [`moveit_benchmark_resources`](https://github.com/moveit/moveit_benchmark_resources/tree/main/moveit_benchmark_resources/bag_files/depth_camera_bag) 中预录制的 3D 点云数据，也可以录制自己的 bag 文件。要录制 bag 文件，首先可以运行 `depth_camera_environment.launch.py` 文件，然后使用以下命令录制 `bag`。

在 `Shell 1` 中运行以下命令：

```bash
ros2 launch moveit2_tutorials depth_camera_environment.launch.py
```

在 `Shell 2` 中运行以下命令：

```bash
ros2 bag record /camera_1/points /camera_1/depth/image_raw /camera_1/depth/camera_info \
                /camera_2/points /camera_2/depth/image_raw /camera_2/depth/camera_info /tf /tf_static
```

当 `Shell 1` 中的命令执行时，会打开一个 `Gazebo` 环境，其中包含两个深度摄像头和一张桌子。此 `Gazebo` 环境用于获取 3D 传感器数据。`Gazebo` 环境应如下图所示。

有关 `depth_camera_environment.launch.py` 的更多说明，您可以查看 GitHub 上的 [`depth_camera_environment.launch.py`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/perception_pipeline/launch/depth_camera_environment.launch.py) 文件中的注释。

对于 Shell 2 中的命令，我们必须保存相机主题和 tf 主题，因为 MoveIt 感知管道需要监听 TF 才能根据世界框架转换点云点的坐标。此外，在 `depth_camera_environment.launch.py` 中发布从 `world` 到摄像头坐标系的静态 tf 的原因是，需要确定机器人与点云之间的变换，MoveIt 的传感器插件稍后会使用此变换。

顺便说一下，您还可以使用 GitHub 上的此 [rviz](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/perception_pipeline/rviz2/depth_camera_environment.rviz) 文件来可视化 rviz 中的点云主题。

在下一步中，我们将使用录制的 bag 文件创建 Octomap。  

## 3D 传感器的配置

**MoveIt 使用基于八叉树的框架来表示周围的世界。以下是 Octomap 的配置参数：**

- **octomap_frame**：指定此表示的坐标系。如果您正在使用移动机器人，此坐标系应为世界中的固定坐标系。我们可以通过点云和图像主题的 `frame_id` 字段为插件设置此坐标系。
- **octomap_resolution**：指定此表示的分辨率（以米为单位）。
- **max_range**：指定此节点处理传感器输入的最大范围值。

现在，当我们播放 bag 文件时，将看到 `/camera_1/points`、`/camera_1/depth/image_raw`、`/camera_1/depth/camera_info`、`/camera_2/points`、`/camera_1/depth/image_raw`、`/camera_1/depth/camera_info`、`/tf` 和 `/tf_static`。我们应该为 MoveIt 创建以下配置文件以处理规划管道中的这些点云主题。您还可以在此处查看 GitHub 上的完整 `sensors_3d.yaml` 配置文件。

`sensors_3d.yaml`：

```yaml
sensors:
  - camera_1_pointcloud
  - camera_2_depth_image

camera_1_pointcloud:
  sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater
  point_cloud_topic: /camera_1/points
  max_range: 5.0
  point_subsample: 1
  padding_offset: 0.1
  padding_scale: 1.0
  max_update_rate: 1.0
  filtered_cloud_topic: /camera_1/filtered_points

camera_2_depth_image:
  sensor_plugin: occupancy_map_monitor/DepthImageOctomapUpdater
  image_topic: /camera_2/depth/image_raw
  queue_size: 5
  near_clipping_plane_distance: 0.3
  far_clipping_plane_distance: 5.0
  shadow_threshold: 0.2
  padding_scale: 1.0
  max_update_rate: 1.0
  filtered_cloud_topic: /camera_2/filtered_points
```

### 点云的配置

通用参数包括：

- **sensor_plugin**：我们使用的插件名称。
- **max_update_rate**：Octomap 表示的更新速率将小于或等于此值。

点云更新器的特定参数包括：

- **point_cloud_topic**：指定监听点云的主题。
- **max_range**（以米为单位）：距离超过此值的点将不会被使用。
- **point_subsample**：每 `point_subsample` 个点中选择一个。
- **padding_offset**：填充的大小（以厘米为单位）。
- **padding_scale**：填充的比例。
- **filtered_cloud_topic**：过滤后的点云将发布到此主题（主要用于调试）。过滤后的点云是执行自过滤后的结果。

## 深度图像的配置

通用参数包括：

- **sensor_plugin**：我们使用的插件名称。
- **max_update_rate**：Octomap 表示的更新速率将小于或等于此值。

深度图更新器的特定参数包括：

- **image_topic**：指定监听深度图像的主题。
- **queue_size**：队列中的图像数量。
- **near_clipping_plane_distance**：可见性的最小距离。
- **far_clipping_plane_distance**：可见性的最大距离。
- **shadow_threshold**：阴影图的最小亮度，低于此值的实体的动态阴影将不可见。
- **padding_offset**：填充的大小（以厘米为单位）。
- **padding_scale**：填充的比例。
- **filtered_cloud_topic**：过滤后的点云将发布到此主题（主要用于调试）。过滤后的点云是执行自过滤后的结果。

## 运行演示

最后一步是运行 `perception_pipeline_demo.launch.py` 并播放我们之前录制的 bag 文件。您可以使用以下命令执行这些步骤。

在 `Shell 3` 中运行：
```bash
ros2 launch moveit2_tutorials perception_pipeline_demo.launch.py
```

在 `Shell 4` 中运行：
```bash
ros2 bag play -r 5 <your_bag_file> --loop
```

[`perception_pipeline_demo.launch.py`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/perception_pipeline/launch/perception_pipeline_demo.launch.py) 类似于 MoveIt 快速入门中的 [`demo.launch.py`](https://github.com/moveit/moveit2_tutorials/blob/main/doc//doc/tutorials/quickstart_in_rviz/launch/demo.launch.py)，除了几个细节。对于 `perception_pipeline_demo.launch.py`，以下行被添加到 `moveit_config` 中。

您可以在 `perception_pipeline_demo.launch.py` 的第 51、52 和 53 行找到这些额外的行：

---

### **第7页**

```python
os.path.join(
    get_package_share_directory("moveit2_tutorials"),
    "config/sensors_3d.yaml"))
```

最后，所有演示代码都可以在 GitHub 上的 [`perception_pipeline`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/perception_pipeline) 目录中找到。

## TroubleShoting

### 问题1

```text
[gzserver-7] Error: Non-unique names detected in <link name='link'>
[gzserver-7]   <collision name='surface'>
[gzserver-7]     <pose>0 0 1 0 -0 0</pose>
```

- 解决
  - 修改`moveit2_tutorials/worlds/perception_pipeline_demo.world`
  - `collision`和`visual`的`name`需要不一样

### 问题2

- 没有相关相机话题，缺失`libgazebo_ros_camera.so`
- 解决

```shell
sudo apt install ros-humble-gazebo-ros-pkgs

# 上面执行完还不行，执行以下命令
sudo apt-get update
sudo apt-get install lsb-release
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt install libgz-plugin2-dev
```


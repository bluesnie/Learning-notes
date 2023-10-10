###### datetime:2023/10/09 11:03

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 2. Cartographer介绍与安装

## 1.Cartographer介绍

![cartographer](imgs/cartographer.gif)

Cartographer是Google开源的一个可跨多个平台和传感器配置以2D和3D形式提供实时同时定位和建图（SLAM）的系统。

> github地址：https://github.com/cartographer-project/cartographer
>
> 文档地址：https://google-cartographer.readthedocs.io/en/latest

机器人里，建图最终方案都是采用了Cartographer，甚至花费大量人力物力对Cartographer算法进行裁剪，这足以表明Cartographer算法的优越性。

> Cartographer系统架构概述（简单看看即可，如果大家后面确定研究方向是SLAM可以深入学习）：
>
> 简单的可以看到左边的可选的输入有深度信息、里程计信息、IMU数据、固定Frame姿态。
>
>
>
> ![系统概述](imgs/high_level_system_overview.png)

## 2.Carttographer安装

### 2.1 apt安装

安装`carotgrapher`

```shell
sudo apt install ros-humble-cartographer
```

需要注意我们不是直接使用`cartographer`，而是通过`cartographer-ros`功能包进行相关操作，所以我们还需要安装下`cartographer-ros`

```shell
sudo apt install ros-humble-cartographer-ros
```

### 2.2 源码安装

比较推荐源码安装的方式，毕竟是以学习为目的，我们后面要稍微看一下源码。

将下面的源码克隆到fishbot_ws的src目录下：

```shell
git clone https://ghproxy.com/https://github.com/ros2/cartographer.git -b ros2
git clone https://ghproxy.com/https://github.com/ros2/cartographer_ros.git -b ros2
```

#### 安装依赖

这里我们使用rosdepc进行依赖的安装，rosdepc指令找不到可以先运行下面的一键安装命令，选择一键配置rosdep即可。

```shell
wget http://fishros.com/install -O fishros && . fishros
```

接着在fishbot_ws下运行下面这个命令进行依赖的安装。

> `rosdepc` 是制作的国内版 `rosdep`，是一个用于安装依赖的工具。该工具的安装可以采用[一键安装](https://fishros.org.cn/forum/topic/20) 进行，选项编号为3。安装完成后运行一次`rodepc update`即可使用。

```shell
rosdepc install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
```

#### 编译

> 这里有一个新的命令`--packages-up-to`，意思是其所有依赖后再编译该包

```shell
colcon build --packages-up-to cartographer_ros
```

### 2.3 测试是否安装成功

如果是源码编译请先source下工作空间后再使用下面指令查看是否安装成功；

```shell
ros2 pkg list | grep cartographer
```

能看到下面的结果即可

```shell
cartographer_ros
cartographer_ros_msgs
```

> 可能你会好奇为什么没有cartographer，因为cartographer包的编译类型原因造成的，不过没关系，cartographer_ros依赖于cartographer，所以有cartographer_ros一定有cartographer。

## 3.Cartographer参数配置

作为一个优秀的开源库，Cartographer提供了很多可以配置的参数，虽然灵活性提高了，但同时也提高了使用难度（需要对参数进行调节配置），所以有必要在正式使用前对参数进行基本的介绍。

因为我们主要使用其进行2D的建图定位，所以我们只需要关注2D相关的参数。

Cartographer参数是使用lua文件来描述的，不会lua也没关系，我们只是改改参数而已。

> 提示：lua中的注释采用 `--` 开头

### 3.1 前端参数

**文件：trajectory_builder_2d**

`src/cartographer/configuration_files/trajectory_builder_2d.lua`

请你打开这个文件自行浏览，对其中我们可能会在初次建图配置的参数进行介绍。

```lua
  -- 是否使用IMU数据
  use_imu_data = true, 
  -- 深度数据最小范围
  min_range = 0.,
  -- 深度数据最大范围
  max_range = 30.,
  -- 传感器数据超出有效范围最大值时，按此值来处理
  missing_data_ray_length = 5.,
  -- 是否使用实时回环检测来进行前端的扫描匹配
  use_online_correlative_scan_matching = true
  -- 运动过滤，检测运动变化，避免机器人静止时插入数据
  motion_filter.max_angle_radians
```

### 3.2 后端参数

**文件：pose_graph.lua**-后端参数配置项

路径`src/cartographer/configuration_files/pose_graph.lua`

该文件主要和地图构建

```lua
--Fast csm的最低分数，高于此分数才进行优化。
constraint_builder.min_score = 0.65
--全局定位最小分数，低于此分数则认为目前全局定位不准确
constraint_builder.global_localization_min_score = 0.7
```

### 3.3 Carotgrapher_ROS参数配置

该部分参数主要是用于和ROS2进行通信和数据收发的配置，比如配置从哪个话题读取里程记数据，从哪个话题来获取深度信息（雷达）。

**文件：backpack_2d.lua**

路径：`src/cartographer_ros/cartographer_ros/configuration_files/backpack_2d.lua`

```lua
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  -- 用来发布子地图的ROS坐标系ID，位姿的父坐标系，通常是map。
  map_frame = "map",
  -- SLAM算法跟随的坐标系ID
  tracking_frame = "base_link",
  -- 将发布map到published_frame之间的tf
  published_frame = "base_link",
  -- 位于“published_frame ”和“map_frame”之间，用来发布本地SLAM结果（非闭环），通常是“odom”
  odom_frame = "odom",
  -- 是否提供里程计
  provide_odom_frame = true,
  -- 只发布二维位姿态（不包含俯仰角）
  publish_frame_projected_to_2d = false,
  -- 是否使用里程计数据
  use_odometry = false,
  -- 是否使用GPS定位
  use_nav_sat = false,
  -- 是否使用路标
  use_landmarks = false,
  -- 订阅的laser scan topics的个数
  num_laser_scans = 0,
  -- 订阅多回波技术laser scan topics的个数
  num_multi_echo_laser_scans = 1,
  -- 分割雷达数据的个数
  num_subdivisions_per_laser_scan = 10,
  -- 订阅的点云topics的个数
  num_point_clouds = 0,
  -- 使用tf2查找变换的超时秒数
  lookup_transform_timeout_sec = 0.2,
  -- 发布submap的周期间隔
  submap_publish_period_sec = 0.3,
  -- 发布姿态的周期间隔
  pose_publish_period_sec = 5e-3,
  -- 轨迹发布周期间隔
  trajectory_publish_period_sec = 30e-3,
  -- 测距仪的采样率
  rangefinder_sampling_ratio = 1.,
  --里程记数据采样率
  odometry_sampling_ratio = 1.,
  -- 固定的frame位姿采样率
  fixed_frame_pose_sampling_ratio = 1.,
  -- IMU数据采样率
  imu_sampling_ratio = 1.,
  -- 路标采样率
  landmarks_sampling_ratio = 1.,
}
```

## 4.总结

本节我们简单的介绍了Cartographer以及二进制和源码安装的方法，并对参数进行介绍。

下一节我们就开始为fishbot配置cartographer，接着就可以使用fishbot进行建图了。

参考文章：

- https://google-cartographer.readthedocs.io/en/latest/configuration.html

--------------

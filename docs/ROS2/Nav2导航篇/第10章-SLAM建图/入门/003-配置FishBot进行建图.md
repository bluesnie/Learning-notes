###### datetime:2023/10/10 11:03

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 3. 配置Fishbot进行建图

上一节我们安装好了cartographer，这节课我们就开始配置cartographer进行建图。

我们需要创建一个功能包，将参数文件和`Cartographer`启动文件放到一起然后启动。

![cartographer](imgs/cartographer_2.gif)

## 1.创建fishbot_cartographer

在src目录下，使用创建功能包指令，创建功能包

```shell
cd src
ros2 pkg create fishbot_cartographer
```

接着创建配置文件夹、launch文件夹和rviz配置文件夹。

```shell
cd fishbot_cartographer
mkdir config
mkdir launch
mkdir rviz
```

创建完成的功能包结构

```
.
├── CMakeLists.txt
├── config
├── launch
├── src
├── package.xml
└── rviz
```

## 2.添加cartographer配置文件

在`fishbot/config`目录下创建`fishbot_2d.lua`文件。

接着我们来写一下配置文件，相较于默认的配置文件，主要修改以下内容（见注释）

```lua
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  -- base_link改为odom,发布map到odom之间的位姿态
  published_frame = "odom",
  odom_frame = "odom",
  -- true改为false，不用提供里程计数据
  provide_odom_frame = false,
  -- false改为true，仅发布2D位资
  publish_frame_projected_to_2d = true,
  -- false改为true，使用里程计数据
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  -- 0改为1,使用一个雷达
  num_laser_scans = 1,
  -- 1改为0，不使用多波雷达
  num_multi_echo_laser_scans = 0,
  -- 10改为1，1/1=1等于不分割
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}


-- false改为true，启动2D SLAM
MAP_BUILDER.use_trajectory_builder_2d = true

-- 0改成0.10,比机器人半径小的都忽略
TRAJECTORY_BUILDER_2D.min_range = 0.10
-- 30改成3.5,限制在雷达最大扫描范围内，越小一般越精确些
TRAJECTORY_BUILDER_2D.max_range = 3.5
-- 5改成3,传感器数据超出有效范围最大值
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.
-- true改成false,不使用IMU数据，大家可以开启，然后对比下效果
TRAJECTORY_BUILDER_2D.use_imu_data = false
-- false改成true,使用实时回环检测来进行前端的扫描匹配
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true 
-- 1.0改成0.1,提高对运动的敏感度
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)

-- 0.55改成0.65,Fast csm的最低分数，高于此分数才进行优化。
POSE_GRAPH.constraint_builder.min_score = 0.65
--0.6改成0.7,全局定位最小分数，低于此分数则认为目前全局定位不准确
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7

-- 设置0可关闭全局SLAM
-- POSE_GRAPH.optimize_every_n_nodes = 0

return options
```

## 3.添加launch文件

### 3.1 launch需要包含的节点

要完成使用Cartographer进行建图，需要两个节点的参与，整个过程的计算流图如下：

![image-20220503212253994](imgs/image-20220503212253994.png)

**/cartographer_node节点:**

该节点从/scan和/odom话题接收数据进行计算，输出/submap_list数据.

该节点需要接收一个参数配置文件（第二部分写的那个）参数。

**/occupancy_grid_node节点：**

该节点接收/submap_list子图列表，然后将其拼接成map并发布

该节点需要配置地图分辨率和更新周期两个参数。

### 3.2 编写launch文件

在路径`src/fishbot_cartographer/launch/`下新建`cartographer.launch.py`文件，接着我们将上面两个节点加入到这个launch文件中。

我们在第二部分写的配置文件就是给cartographer_node节点的，可以通过这个节点启动参数`configuration_directory`和`configuration_basename`进行传递。

```

import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 定位到功能包的地址
    pkg_share = FindPackageShare(package='fishbot_cartographer').find('fishbot_cartographer')
    
    #=====================运行节点需要的配置=======================================================================
    # 是否使用仿真时间，我们用gazebo，这里设置成true
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    # 地图的分辨率
    resolution = LaunchConfiguration('resolution', default='0.05')
    # 地图的发布周期
    publish_period_sec = LaunchConfiguration('publish_period_sec', default='1.0')
    # 配置文件夹路径
    configuration_directory = LaunchConfiguration('configuration_directory',default= os.path.join(pkg_share, 'config') )
    # 配置文件
    configuration_basename = LaunchConfiguration('configuration_basename', default='fishbot_2d.lua')
    rviz_config_dir = os.path.join(pkg_share, 'config')+"/cartographer.rviz"
    print(f"rviz config in {rviz_config_dir}")

    
    #=====================声明三个节点，cartographer/occupancy_grid_node/rviz_node=================================
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-configuration_directory', configuration_directory,
                   '-configuration_basename', configuration_basename])

    cartographer_occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', resolution, '-publish_period_sec', publish_period_sec])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    #===============================================定义启动文件========================================================
    ld = LaunchDescription()
    ld.add_action(cartographer_node)
    ld.add_action(cartographer_occupancy_grid_node)
    ld.add_action(rviz_node)

    return ld
```

## 4.添加安装指令

做完上面的操作，我们还需要添加安装指令。

打开CmakeLists.txt，添加下面一条指令，将三个目录安装到install目录。

```
install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)
```

## 5.开始建图

### 5.1编译启动

```shell
colcon build --packages-select fishbot_cartographer 
```

启动建图前，需要先启动gazebo仿真环境，因为我们的建图程序依赖于Gazebo提供雷达和里程计等数据。

```shell
source install/setup.bash
ros2 launch fishbot_description gazebo.launch.py
```

source,启动建图

```shell
source install/setup.bash
ros2 launch fishbot_cartographer cartographer.launch.py 
```

### 5.2修改配置

如果一切正常，你应该看到的是一个空空如也的RVIZ界面

![image-20220503221911478](imgs/image-20220503221911478.png)

不用担心，此时地图其实已经有了，我们需要添加一下地图相关的插件即可。

通过`Add`->`By Topic`添加组件。

![image-20220503222230429](imgs/image-20220503222230429.png)

最后通过左边的插件你应该可以看到图和机器人了。

![image-20220503222158266](imgs/image-20220503222158266.png)

### 5.3开始建图

打开我们机器人遥控节点，降低速度，控制机器人走一圈，看看地图的变化。

```
ros2 run teleop_twist_keyboard teleop_twist_keyboard 
```

![cartographer](imgs/cartographer_2.gif)

### 6.保存地图

走完一圈，没有黑影部分，我们就可以保存地图为一个本地文件了。我们需要使用一个叫做`nav2_map_server`的工具。

### 6.1 安装nav2_map_server

```
sudo apt install ros-humble-nav2-map-server
```

### 6.2 保存地图

```
ros2 run nav2_map_server map_saver_cli --help
```

可以看到有下面的用法

```shell
Usage:
  map_saver_cli [arguments] [--ros-args ROS remapping args]

Arguments:
  -h/--help
  -t <map_topic>
  -f <mapname>
  --occ <threshold_occupied>
  --free <threshold_free>
  --fmt <image_format>
  --mode trinary(default)/scale/raw

NOTE: --ros-args should be passed at the end of command line
```

我们的地图话题为map，文件名字我们用fishbot_map,所以有下面这个这样写的命令行。

```shell
# 先将地图保存到src/fishbot_cartographer/map目录下
cd src/fishbot_cartographer/ && mkdir map && cd map
ros2 run nav2_map_server map_saver_cli -t map -f fishbot_map
```

接着我们就可以得到下面的两个文件

```shell
.
├── fishbot_map.pgm
└── fishbot_map.yaml

0 directories, 2 files
```

这两个文件就是对当前地图保存下来的文件，其中.pgm是地图的数据文件，.yaml后缀的是地图的描述文件。

下面的导航过程中我们将要使用到地图文件进行路径的搜索和规划。

## 7.总结

本节带你一起完成了Fishbot的Cartographer配置和建图，下一节会对地图是什么以及如何使用进行介绍，接着我们就可以配置Nav2进行Fishbot的导航了。



--------------

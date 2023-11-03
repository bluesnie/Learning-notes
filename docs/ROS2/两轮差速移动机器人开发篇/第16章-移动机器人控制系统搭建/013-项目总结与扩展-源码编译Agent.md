###### datetime:2023/11/03 10:52

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 16.项目总结与扩展

上一节我们完成后，就可以通过话题获取到机器人的里程计，也可以通过话题控制机器人移动了。对于一个移动底盘来说，这两个话题就是最关键的两个，有了这两个就可以控制底盘完成移动。但如果想让底盘更好用，可以增加OLED模块显示数据，比如电池的电压。同时我们的板子还支持IMU和超声波模块，其实都可以通过话题发布出来。

说了那么多，你可能不知道怎么做，没关系，小鱼大佬写好了所有的代码，有了前面的基础，加上代码中的详细注释，看懂他们对你来说并不难。完成代码可以直接通过git进行克隆到本地。

```shell
git clone http://github.fishros.org/https://github.com/fishros/fishbot_motion_control_microros.git
```

如果你的网络不错，也可以在线查看代码：https://github.dev/fishros/fishbot_motion_control_microros

该代码的结构如下：

```
.
├── extra_packages
│   └── fishbot_interfaces
│       ├── CMakeLists.txt   # CMake构建配置文件
│       ├── msg
│       │   └── MyCustomMessage.msg   # ROS消息定义
│       ├── package.xml   # ROS包描述文件
│       └── srv
│           └── FishBotConfig.srv   # ROS服务定义
├── include
│   ├── fishbot_config.h   # 头文件
│   ├── fishbot.h   # 头文件
│   ├── fishlog.h   # 头文件
│   └── README   # 说明文档
├── Installer   # 安装器相关
├── lib
│   ├── Displays
│   │   ├── fishbot_display.cpp   # 显示相关源码
│   │   └── fishbot_display.h   # 显示相关头文件
│   ├── FishbotUtils
│   │   ├── fishbot_utils.cpp   # 实用工具源码
│   │   └── fishbot_utils.h   # 实用工具头文件
│   ├── Kinematics
│   │   ├── Kinematics.cpp   # 运动学计算源码
│   │   └── Kinematics.h   # 运动学计算头文件
│   ├── MicroRosRwm
│   │   ├── micro_ros_transport_serial.cpp   # MicroROS串口传输源码
│   │   ├── micro_ros_transport_serial.h   # MicroROS串口传输头文件
│   │   ├── micro_ros_transport_wifi_udp.cpp   # MicroROS WiFi/UDP传输源码
│   │   └── micro_ros_transport_wifi_udp.h   # MicroROS WiFi/UDP传输头文件
│   ├── PidController
│   │   ├── PidController.cpp   # PID控制器源码
│   │   └── PidController.h   # PID控制器头文件
│   └── README   # 说明文档
├── LICENSE   # 许可证文件
├── partition.csv   # 分区配置文件
├── platformio.ini   # PlatformIO配置文件
├── README.md   # 项目主README文件
├── RELEASES.md   # 发布说明
├── src
    ├── fishbot_config.cpp   # 配置相关源码
    ├── fishbot.cpp   # 主要功能源码
    └── main.cpp   # 主程序入口源码
```

下一章我们开始学习如何进行导航和建图。

# 17.拓展-源码编译Agent

本文介绍了如何拓展MicroROS的Agent，将其作为一个功能包进行源码编译，并提供了详细的步骤如下：

## 一、下载microros-agent

首先，我们需要下载MicroROS的Agent源码，并准备相应的依赖。以下是下载和准备的步骤：

1. 安装必要的依赖项：

   ```
   sudo apt-get install -y build-essential
   ```

2. 创建工作空间并进入源码目录：

   ```
   mkdir -p microros_ws/src
   cd microros_ws/src
   ```

3. 下载MicroROS Agent和相关消息包的源码：

   ```
   git clone http://github.fishros.org/https://github.com/micro-ROS/micro-ROS-Agent.git -b humble
   git clone http://github.fishros.org/https://github.com/micro-ROS/micro_ros_msgs.git -b humble
   ```

## 二、编译运行

在成功下载源码并准备好依赖后，我们可以进行编译并运行MicroROS Agent。以下是编译和运行的步骤：

1. 返回工作空间目录并执行编译：

   ```
   cd microros_ws
   colcon build
   ```

2. 运行MicroROS
   Agent，注意可能存在串口权限问题。可以[参考链接](https://fishros.org.cn/forum/topic/1150) 来设置权限。运行命令如下（假设串口为`/dev/ttyUSB0`，波特率为921600）：

   ```shell
   ros2 run micro_ros_agent micro_ros_agent serial -b 921600 --dev /dev/ttyUSB0 -v
   ```

   或者使用UDP方法：

   ```shell
   ros2 run micro_ros_agent micro_ros_agent serial udp4 --port 8888 -v6
   ```

## 三、总结

通过上述步骤，我们成功地拓展了MicroROS的Agent功能包，实现了源码的编译和运行。通过MicroROS Agent，我们能够在资源受限的嵌入式系统中实现强大的ROS 2通信能力。
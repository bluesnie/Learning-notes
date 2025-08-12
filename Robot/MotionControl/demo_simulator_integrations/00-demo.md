###### datetime:2025/06/26 10:34

###### author:nzb

# Demo

这个 `GitHub` 仓库提供了支持 [`ros2_control`](https://github.com/ros-controls/ros2_control_demos) 功能的机器人和简单模拟的模板，用于展示和验证 `ros2_control` 概念。

如果你想了解如何使用 `ros2_control` 的详细步骤，可以查看 [`ros-control/roscon2022_workshop`](https://github.com/ros-controls/roscon2022_workshop) 仓库。

## 本仓库包含以下内容

本仓库展示了以下 `ros2_control` 概念：

- 为系统、传感器和执行器创建 `HardwareInterface`。
- 以 `URDF` 文件形式创建机器人描述。
- 加载配置并使用启动文件启动机器人。
- 差分移动底盘 `DiffBot` 的控制。
- `RRBot` 的两个关节控制。
- 六自由度机器人的控制。
- 为机器人实现控制器切换策略。
- 在 `ros2_control` 中使用关节限位和传动概念。

## [安装](https://control.ros.org/humble/doc/ros2_control_demos/doc/index.html#installation)

- 执行`rosdep install --from-paths src --ignore-src -r -y`安装报错的时候，更新下 `humble` 相关的包
- 使用源码编译，`ros2_control_demos` 使用 `humble` 分支

报错信息

```text
Fetched 5,475 B in 4s (1,533 B/s)
Reading package lists... Done
W: http://dl.winehq.org/wine-builds/ubuntu/dists/jammy/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
W: http://packages.osrfoundation.org/gazebo/ubuntu-stable/dists/jammy/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://typora.io/linux ./ InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY BA300B7755AFCFAE
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: http://packages.ros.org/ros2/ubuntu jammy InRelease: The following signatures were invalid: EXPKEYSIG F42ED6FBAB17C654 Open Robotics <info@osrfoundation.org>
W: Failed to fetch https://typora.io/linux/./InRelease  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY BA300B7755AFCFAE
W: Failed to fetch http://packages.ros.org/ros2/ubuntu/dists/jammy/InRelease  The following signatures were invalid: EXPKEYSIG F42ED6FBAB17C654 Open Robotics <info@osrfoundation.org>
W: Some index files failed to download. They have been ignored, or old ones used instead.
W: Target Packages (Packages) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_typora_io_linux-jammy.list:1 and /etc/apt/sources.list.d/typora.list:1
W: Target Translations (en_US) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_typora_io_linux-jammy.list:1 and /etc/apt/sources.list.d/typora.list:1
W: Target Translations (en) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_typora_io_linux-jammy.list:1 and /etc/apt/sources.list.d/typora.list:1
W: Target Translations (zh_CN) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_typora_io_linux-jammy.list:1 and /etc/apt/sources.list.d/typora.list:1
```

更新

```bash
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" # If using Ubuntu derivates use $UBUNTU_CODENAME
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt upgrade
```

## 目标

该仓库有两个其他目标：

- 实现了 `ros-controls/roadmap` 仓库文件 `components_architecture_and_urdf_examples` 中描述的示例配置。
- 该仓库是 `ros2_control` 概念的验证环境，这些概念只能在运行时进行测试（例如，控制器管理器执行控制器、机器人硬件与控制器之间的通信）。

# 示例概述

## [Example 1: RRBot](https://control.ros.org/humble/doc/ros2_control_demos/example_1/doc/userdoc.html#example-1-rrbot)

`RRBot` - 或称为“转动-转动机械臂机器人” - 是一个具有一个硬件接口的简单位置控制机器人。这个示例还展示了在不同控制器之间切换的过程。

## [Example 2: DiffBot](https://control.ros.org/humble/doc/ros2_control_demos/example_2/doc/userdoc.html#diffbot)

`DiffBot`，或称为“差分移动机器人”，是一个具有差分驱动的简单移动平台。这个机器人基本上是一个根据差分驱动运动学原理移动的盒子。

## [Example 3: “RRBot with multiple interfaces”](https://control.ros.org/humble/doc/ros2_control_demos/example_3/doc/userdoc.html#example-3-robots-with-multiple-interfaces)

带多个接口的 RRBot。

## [Example 4: “Industrial robot with integrated sensor”](https://control.ros.org/humble/doc/ros2_control_demos/example_4/doc/userdoc.html#example-4-industrial-robot-with-integrated-sensor)

带集成传感器的工业机器人

## [Example 5: “Industrial robot with externally connected sensor”](https://control.ros.org/humble/doc/ros2_control_demos/example_5/doc/userdoc.html#example-5-industrial-robot-with-externally-connected-sensor)

工业机器人，外部连接传感器

## [Example 6: “Modular robot with separate communication to each actuator”](https://control.ros.org/humble/doc/ros2_control_demos/example_6/doc/userdoc.html#example-6-modular-robots-with-separate-communication-to-each-actuator)

该示例展示了如何实现每个执行器有独立通信的机器人硬件（一个关节一个执行器（硬件插件））。

## [Example 7: “6-DOF robot”](https://control.ros.org/humble/doc/ros2_control_demos/example_7/doc/userdoc.html#example-7-full-tutorial-with-a-6dof-robot)

一个面向中级 `ROS2` 用户的 6 自由度机器人完整教程。

## [Example 8: “Using transmissions”](https://control.ros.org/humble/doc/ros2_control_demos/example_8/doc/userdoc.html#example-8-industrial-robots-with-an-exposed-transmission-interface)

带有暴露传动接口的 `RRBot`。

## [Example 9: “Gazebo Classic”](https://control.ros.org/humble/doc/ros2_control_demos/example_9/doc/userdoc.html#example-9-simulation-with-rrbot)

演示如何在模拟和硬件之间切换。

## [Example 10: “GPIO interfaces”](https://control.ros.org/humble/doc/ros2_control_demos/example_10/doc/userdoc.html#example-10-industrial-robot-with-gpio-interfaces)

工业机器人带 GPIO 接口

## [Example 11: “CarlikeBot”](https://control.ros.org/humble/doc/ros2_control_demos/example_11/doc/userdoc.html#carlikebot)

带自行车转向控制器的 `CarlikeBot`

## [Example 12: “Controller chaining”](https://control.ros.org/humble/doc/ros2_control_demos/example_12/doc/userdoc.html#example-12-controller-chaining-with-rrbot)

示例展示了一个简单的可链式控制器及其集成，形成控制器链来控制 `RRBot` 的关节。

## [Example 13: “Multi-robot system with hardware lifecycle management”](https://control.ros.org/humble/doc/ros2_control_demos/example_13/doc/userdoc.html#example-13-multi-robot-system-with-lifecycle-management)

此示例展示了如何在单个控制器管理器实例中处理多个机器人。

## [Example 14: “Modular robots with actuators not providing states and with additional sensors”](https://control.ros.org/humble/doc/ros2_control_demos/example_14/doc/userdoc.html#example-14-modular-robot-with-actuators-not-providing-states)

示例展示了如何实现不提供状态的执行器和带有额外传感器的机器人硬件。

## [Example 15: “Using multiple controller managers”](https://control.ros.org/humble/doc/ros2_control_demos/example_15/doc/userdoc.html#example-15-using-multiple-controller-managers)

本示例展示了如何在不同的控制器管理器实例下集成多台机器人。

# 命令

- `ros2 control list_controllers`
- `ros2 control list_controller_types`
- `ros2 control list_hardware_components`
- `ros2 control list_hardware_interfaces`
- `ros2 control load_controller`
- `ros2 control reload_controller_libraries`
- `ros2 control set_controller_state`
- `ros2 control set_hardware_component_state`
- `ros2 control switch_controllers`
- `ros2 control unload_controller`
- `ros2 control view_controller_chains`
###### datetime:2025/07/09 10:50

###### author:nzb

# [gz_ros2_control](https://control.ros.org/humble/doc/gz_ros2_control/doc/index.html#ign-ros2-control)

这是一个用于将 `ros2_control` 控制器架构与 `Gazebo` 模拟器集成的 `ROS2` 软件包。该软件包提供了一个 `Gazebo-Sim` 系统插件，该插件实例化一个 `ros2_control` 控制器管理器并将其连接到 `Gazebo` 模型。

## 安装

### 二进制软件包

`gz_ros2_control` 为 `ROS2 humble` 在 `Ubuntu` 上发布。要使用它，您必须安装 `ros-humble-gz-ros2-control` 软件包，例如，通过运行以下命令：`sudo apt install ros-humble-gz-ros2-control ros-humble-gz-ros2-control-demos`

### [从源代码构建](https://control.ros.org/humble/doc/gz_ros2_control/doc/index.html#building-from-source)

### [使用 docker](https://control.ros.org/humble/doc/gz_ros2_control/doc/index.html#using-docker)

## 为 URDF 添加 ros2_control 标签

### 简单设置 

要使用 `ros2_control` 与您的机器人一起工作，您需要在您的 URDF 中添加一些额外的元素。您应该包含标签 `<ros2_control>` 以访问和控制机器人接口。我们应该包含：

- 我们机器人的特定 `<plugin>` 标签
- `<joint>` 标签包括机器人控制器：命令和状态。

```xml
<ros2_control name="GazeboSimSystem" type="system">
  <hardware>
    <plugin>gz_ros2_control/GazeboSimSystem</plugin>
  </hardware>
  <joint name="slider_to_cart">
    <command_interface name="effort">
      <param name="min">-1000</param>
      <param name="max">1000</param>
    </command_interface>
    <state_interface name="position">
      <param name="initial_value">1.0</param>
    </state_interface>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>
</ros2_control>
```

### 在仿真中使用 mimic 关节

要在 `gz_ros2_control` 中使用 `mimic` 关节，您需要将其参数定义到您的 `URDF` 中。我们应该包含：

- `<mimic>` 标签到模拟关节的[详细手册](https://wiki.ros.org/urdf/XML/joint)
- `mimic` 和 `multiplier` 参数到 `<ros2_control>` 标签中的关节定义

```xml
<joint name="left_finger_joint" type="prismatic">
  <mimic joint="right_finger_joint"/>
  <axis xyz="0 1 0"/>
  <origin xyz="0.0 0.48 1" rpy="0.0 0.0 3.1415926535"/>
  <parent link="base"/>
  <child link="finger_left"/>
  <limit effort="1000.0" lower="0" upper="0.38" velocity="10"/>
</joint>
```

```xml
<joint name="left_finger_joint">
  <param name="mimic">right_finger_joint</param>
  <param name="multiplier">1</param>
  <command_interface name="position"/>
  <state_interface name="position"/>
  <state_interface name="velocity"/>
  <state_interface name="effort"/>
</joint>
```

## 添加 gz_ros2_control 插件

除了 `ros2_control` 标签外，您的 `URDF` 还需要添加一个 `Gazebo` 插件，该插件实际解析 `ros2_control` `标签并加载相应的硬件接口和控制器管理器。默认情况下，gz_ros2_control` 插件非常简单，但它也可以通过额外的插件架构进行扩展，允许高级用户在 `ros2_control` 和 `Gazebo` 之间创建自定义的机器人硬件接口。

```xml
<gazebo>
  <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
    <robot_param>robot_description</robot_param>
    <robot_param_node>robot_state_publisher</robot_param_node>
    <parameters>$(find gz_ros2_control_demos)/config/cart_controller.yaml</parameters>
  </plugin>
</gazebo>
```

`gz_ros2_control` 的 `<plugin>` 标签还具有以下可选的子元素：

- `<robot_param>` : `robot_description` （`URDF`）在参数服务器上的位置，默认为 `robot_description`
- `<robot_param_node>` : `robot_param` 所在节点的名称，默认为 `robot_state_publisher`
- `<parameters>` : 一个包含控制器配置的 `YAML` 文件。此元素可以多次给出以加载多个文件。
- `<controller_manager_name>` : 设置控制器管理器名称（默认： `controller_manager` ）

此外，还可以指定命名空间和重映射规则，这些规则将传递给控制器管理器并加载的控制器。添加以下 `<ros>` 部分：

```xml
<gazebo>
  <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
    ...
    <ros>
      <namespace>my_namespace</namespace>
      <remapping>/robot_description:=/robot_description_full</remapping>
    </ros>
  </plugin>
</gazebo>
```

### 默认 gz_ros2_control 行为

默认情况下，如果没有 `<plugin>` 标签，`gz_ros2_control` 将尝试从 `URDF` 中获取所有与基于 `ros2_control` 的控制器接口所需的信息。这对于大多数情况来说是足够的，并且至少可以让你开始使用。

默认行为提供以下 `ros2_control` 接口：
- `hardware_interface::JointStateInterface`
- `hardware_interface::EffortJointInterface`
- `hardware_interface::VelocityJointInterface`

### 高级：自定义 gz_ros2_control 模拟插件

`gz_ros2_control` `Gazebo` 插件还提供了一个基于 `pluginlib` 的接口，用于在 `Gazebo` 和 `ros2_control` 之间实现自定义接口，以模拟更复杂的机构（如非线性弹簧、连杆等）。


这些插件必须继承 `gz_ros2_control::GazeboSimSystemInterface` ，它实现了一个模拟的 `ros2_control` `hardware_interface::SystemInterface` 。`SystemInterface` 提供了 `API` 级别的访问权限，用于读取和命令关节属性。

相应的 `GazeboSimSystemInterface` 子类在 `URDF` 模型中指定，并在加载机器人模型时加载。例如，以下 `XML` 将加载默认插件：

```xml
<ros2_control name="GazeboSimSystem" type="system">
  <hardware>
    <plugin>gz_ros2_control/GazeboSimSystem</plugin>
  </hardware>
  ...
<ros2_control>
<gazebo>
  <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
    ...
  </plugin>
</gazebo>
```

### 设置控制器

在 `<plugin>` 中使用标签 `<parameters>` 来设置包含控制器配置的 `YAML` 文件，并使用标签 `<controller_manager_name>` 来设置控制器管理器节点的名称。

以下是一个控制器的基本配置：

- `joint_state_broadcaster` ：此控制器将注册到 `hardware_interface::StateInterface` 的所有资源的状态发布到类型为 `sensor_msgs/msg/JointState` 的主题上。
- `joint_trajectory_controller`：此控制器创建一个名为 `/joint_trajectory_controller/follow_joint_trajectory` 的动作，类型为 `control_msgs::action::FollowJointTrajectory` 。

```yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

joint_trajectory_controller:
  ros__parameters:
    joints:
      - slider_to_cart
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
```

## gz_ros2_control_demos

[`gz_ros2_control_demos`](https://github.com/ros-controls/gz_ros2_control) 包中包含一些示例。

### 轨道上的小车

这些示例允许在一个 30 米长的轨道上启动小车。

```shell
# 启动 gazebo
ros2 launch gz_ros2_control_demos cart_example_position.launch.py
ros2 launch gz_ros2_control_demos cart_example_velocity.launch.py
ros2 launch gz_ros2_control_demos cart_example_effort.launch.py

# 启动控制器
ros2 run gz_ros2_control_demos example_position
ros2 run gz_ros2_control_demos example_velocity
ros2 run gz_ros2_control_demos example_effort
```

## 移动机器人

```shell
# 启动 gazebo
ros2 launch gz_ros2_control_demos diff_drive_example.launch.py
ros2 launch gz_ros2_control_demos tricycle_drive_example.launch.py
ros2 launch gz_ros2_control_demos ackermann_drive_example.launch.py

# 启动控制器
ros2 run gz_ros2_control_demos example_diff_drive
ros2 run gz_ros2_control_demos example_tricycle_drive
ros2 run gz_ros2_control_demos example_ackermann_drive
```

为了演示命名空间机器人的设置，运行

```shell
# 这将启动一个位于命名空间 r1 的差速驱动机器人。
ros2 launch gz_ros2_control_demos diff_drive_example_namespaced.launch.py
```

> `controller_manager` 的 `ros2_control` 设置在 `diff_drive_controller.yaml` 中定义的控制器使用通配符来匹配所有命名空间。

要运行万向轮移动机器人，请运行以下命令从键盘控制它：

```shell
ros2 launch gz_ros2_control_demos mecanum_drive_example.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

## 夹爪

```shell
# 以下示例展示了一个带有模仿关节的平行夹爪
ros2 launch gz_ros2_control_demos gripper_mimic_joint_example.launch.py
# 发送示例命令
ros2 run gz_ros2_control_demos example_gripper
```

## 带被动关节的摆杆（小车-摆杆）

```shell
# 以下示例展示了一个带有摆杆臂的小车
ros2 launch gz_ros2_control_demos pendulum_example_effort.launch.py
ros2 run gz_ros2_control_demos example_effort

# 这使用了小车在轨道上的自由度上的力矩指令接口。为了证明即使使用位置指令接口，摆杆的被动关节的物理问题也能正确解决，请运行
ros2 launch gz_ros2_control_demos pendulum_example_position.launch.py
ros2 run gz_ros2_control_demos example_position
```

## 社区

下面列出了支持与 `ros2_control` 集成的模拟器。要添加您的 `ros2_control` 集成，请向 `Github` 上的此页面提交 `PR！`

- [`Isaac Sim`](https://moveit.picknik.ai/main/doc/how_to_guides/isaac_panda/isaac_panda_tutorial.html)
- [`Webots`](https://github.com/cyberbotics/webots_ros2/tree/master/webots_ros2_control)
- [`针对具有基于主题系统的模拟器的自定义接口`](https://github.com/PickNikRobotics/topic_based_ros2_control)
- [`MuJoCo`](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers/tree/ros2/cartesian_controller_simulation)
- [`Algoryx AGX Dynamic`](https://github.com/Algoryx/agx-ros2-collection)

## Gazebo Troubleshooting

```text
[gazebo-2] [Wrn] [ModelDatabase.cc:340] Getting models from[http://models.gazebosim.org/]. This may take a few seconds.
[gazebo-2] [Wrn] [ModelDatabase.cc:212] Unable to connect to model database using [http://models.gazebosim.org//database.config]. Only locally installed models will be available.
```

- 解决方法

```bash
cd /usr/share/gazebo-11/models/
wget http://file.ncnynl.com/ros/gazebo_models.txt

wget -i gazebo_models.txt
ls model.tar.g* | xargs -n1 tar xzvf
```

- 使用Gazebo环境

~~- 激活环境: `source /usr/share/gazebo-11/setup.sh`~~

```text
[ign gazebo-1] [Err] [SystemPaths.cc:378] Unable to find file with URI [model://cyan_bot_gz_sim/urdf/meshes/base_link_cls.STL]
[ign gazebo-1] [Err] [SystemPaths.cc:473] Could not resolve file [model://cyan_bot_gz_sim/urdf/meshes/base_link_cls.STL]
[ign gazebo-1] [Err] [MeshManager.cc:173] Unable to find file[model://cyan_bot_gz_sim/urdf/meshes/base_link_cls.STL]
```

- 添加自定义模型路径: 
  - 旧版`gazebo`：`export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(ros2 pkg prefix cyan_bot_gz_sim)/share`
  - 新版`gazebo`：`export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$(ros2 pkg prefix cyan_bot_gz_sim)/share`
- 启动：
  - `source /opt/ros/humble/setup.bash`
  - `source install/setup.bash`
  - `ros2 launch cyan_bot_gz_sim orca_gazebo.launch.py` 
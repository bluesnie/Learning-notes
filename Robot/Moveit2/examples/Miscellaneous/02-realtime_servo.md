###### datetime:2025/04/10 16:17

###### author:nzb

# 实时伺服

`MoveIt Servo` 可实现对机械臂的实时控制。

`MoveIt Servo`可接受以下任何类型的命令：

- 单个关节速度
- 末端效应器的预期速度
- 末端执行器的预期姿势。

这样就能通过多种输入方案实现遥控操作，或通过其他自主软件控制机器人，例如视觉伺服或闭环位置控制。

## 设计概述

`MoveIt Servo` 由两个主要部分组成： 核心实现 `Servo`，它提供了一个 C++ 接口, 和伺服节点（`ServoNode`），后者封装了 C++ 接口，并提供了一个 ROS 接口。伺服系统的配置通过 `servo_parameters.yaml` 中指定的 ROS 参数完成

除了伺服功能外，`MoveIt Servo` 还具有一些便利功能，例如

- 奇异点检查
- 碰撞检查
- 运动平滑
- 执行关节位置和速度限制

奇点检查和碰撞检查是一种安全功能，可在接近奇点或碰撞（自身碰撞或与其他物体碰撞）时降低速度。碰撞检查和平滑是可选功能，可分别使用 `check_collisions` 参数和 `use_smoothing` 参数禁用。

如果提供了逆运动学求解器，则可通过雅可比或机器人的 `IK` 求解器进行处理。

### 伺服中的逆运动学

逆运动学可由 `MoveIt Servo` 通过逆雅可比计算进行内部处理。不过，您也可以使用 `IK` 插件。要在 `MoveIt Servo` 中配置 `IK` 插件，您的机器人配置包必须在 `kinematics.yaml` 文件（如 `Panda` 配置包中的文件）中定义一个。`MoveIt` 内部和外部都有多个IK插件，其中 `bio_ik/BioIKKinematicsPlugin` 是最常见的选择。

填充好 `kinematics.yaml` 文件后，将其与传递给伺服节点的 `ROS` 参数一起包含在启动文件中：

```python
moveit_config = (
    MoveItConfigsBuilder("moveit_resources_panda")
    .robot_description(file_path="config/panda.urdf.xacro")
    .to_moveit_configs()
)
servo_node = Node(
    package="moveit_servo",
    executable="servo_node",
    parameters=[
        servo_params,
        low_pass_filter_coeff,
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.robot_description_kinematics, # here is where kinematics plugin parameters are passed
    ],
)
```

以上摘录自 `MoveIt` 中的 [`servo_example.launch.py`](https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/launch/demo_ros_api.launch.py)。在上述示例中，`kinematics.yaml` 文件取自工作区中的 [`moveit_resources`](https://github.com/moveit/moveit_resources/tree/ros2//) 资源库，特别是 `moveit_resources/panda_moveit_config/config/kinematics.yaml`。通过加载 `yaml` 文件传递的实际 `ROS` 参数名的形式为 `robot_description_kinematics.<group_name>.<param_name>`，例如 `robot_description_kinematics.panda_arm.kinematics_solver`。

由于 `moveit_servo` 不允许在伺服节点上设置在 `kinematics.yaml` 文件中找到的未声明参数，因此需要在插件代码中声明自定义求解器参数。

例如，`bio_ik` 在 [`bio_ik/src/kinematics_plugin.cpp`](https://github.com/PickNikRobotics/bio_ik/blob/ros2/src/kinematics_plugin.cpp#L160) 中定义了一个 `getROSParam()` 函数，用于在伺服节点上找不到参数时声明参数。

## 主题优先权

为了在控制硬件时获得最佳性能，您希望主伺服回路的抖动越小越好。普通的 `linux` 内核针对计算吞吐量进行了优化，因此不太适合硬件控制。两种最简单的内核选择是 `Ubuntu 22.04 LTS Beta` 版的实时内核或 `Debian Bullseye `上的 `linux-image-rt-amd64`。

如果安装了实时内核，`ServoNode` 的主线程会自动尝试配置优先级为 `40` 的 `SCHED_FIFO`。请参阅 [`config/servo_parameters.yaml`](https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/config/servo_parameters.yaml) 中的更多文档。

## 在新机器人上进行设置

在机器人上运行 `MoveIt Servo` 的最低要求包括
- 有效的机器人 URDF 和 SRDF。
- 可接受关节位置或速度的控制器。
- 可提供快速、准确关节位置反馈的关节编码器。
由于运动学是由 `MoveIt` 的核心部分处理的，因此建议您为自己的机器人准备一个有效的配置包，并运行其中包含的演示启动文件。

## 使用 C++ 接口

当需要避免 `ROS` 通信基础设施开销的性能要求时，或者当伺服生成的输出需要输入到没有 `ROS` 接口的其他控制器时，这样做会很有好处。

当使用 C++ 接口的 `MoveIt Servo` 时，三种输入命令类型分别为 `JointJogCommand`、`TwistCommand` 和 `PoseCommand`。使用 C++ 接口时，`Servo` 的输出是 `KinematicState`（运动状态），这是一个包含关节名称、位置、速度和加速度的结构。如数据类型头文件中的[定义](https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/include/moveit_servo/utils/datatypes.hpp)所示。

第一步是创建一个伺服实例。

```cpp
// Import the Servo headers.
#include <moveit_servo/servo.hpp>
#include <moveit_servo/utils/common.hpp>

// The node to be used by Servo.
rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("servo_tutorial");

// Get the Servo parameters.
const std::string param_namespace = "moveit_servo";
const std::shared_ptr<const servo::ParamListener> servo_param_listener =
    std::make_shared<const servo::ParamListener>(node, param_namespace);
const servo::Params servo_params = servo_param_listener->get_params();

// Create the planning scene monitor.
const planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor =
    createPlanningSceneMonitor(node, servo_params);

// Create a Servo instance.
Servo servo = Servo(node, servo_param_listener, planning_scene_monitor);
```

使用 `JointJogCommand`

```cpp
using namespace moveit_servo;

// Create the command.
JointJogCommand command;
command.joint_names = {"panda_link7"};
command.velocities = {0.1};

// Set JointJogCommand as the input type.
servo.setCommandType(CommandType::JOINT_JOG);

// Get the joint states required to follow the command.
// This is generally run in a loop.
KinematicState next_joint_state = servo.getNextJointState(command);
```

使用 `TwistCommand`

```cpp
using namespace moveit_servo;

// Create the command.
TwistCommand command{"panda_link0", {0.1, 0.0, 0.0, 0.0, 0.0, 0.0};

// Set the command type.
servo.setCommandType(CommandType::TWIST);

// Get the joint states required to follow the command.
// This is generally run in a loop.
KinematicState next_joint_state = servo.getNextJointState(command);
```

使用 `PoseCommand`
```cpp
using namespace moveit_servo;

// Create the command.
Eigen::Isometry3d ee_pose = Eigen::Isometry3d::Identity(); // This is a dummy pose.
PoseCommand command{"panda_link0", ee_pose};

// Set the command type.
servo.setCommandType(CommandType::POSE);

// Get the joint states required to follow the command.
// This is generally run in a loop.
KinematicState next_joint_state = servo.getNextJointState(command);
```

然后，`next_joint_state` 的结果可用于控制流水线的后续步骤。

最后一条命令所产生的 `MoveIt` 伺服器状态可通过以下方式获得：

```cpp
StatusCode status = servo.getStatus();
```

用户可利用状态进行更高层次的决策。

有关使用 C++ 界面的完整示例，请参见 `moveit_servo/demos`。可以使用 `moveit_servo/launch` 中的启动文件[启动演示](https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/launch)程序。 

```bash
ros2 launch moveit_servo demo_joint_jog.launch.py
ros2 launch moveit_servo demo_twist.launch.py
ros2 launch moveit_servo demo_pose.launch.py
```

## 使用 ROS 接口

要通过 `ROS` 接口使用 `MoveIt Servo`，必须将其作为节点或组件启动，并提供所需的参数，如[示例](https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/launch/demo_ros_api.launch.py)所示。

在 `ROS` 接口上使用` MoveIt Servo` 时，命令是发布到伺服参数指定的各自主题中的下列类型的 `ROS` 信息。

- `control_msgs::msg::JointJog` 由 `joint_command_in_topic` 参数指定的主题。
- `geometry_msgs::msg::TwistStamped`内容为 `cartesian_command_in_topic` 参数指定的主题。目前，扭转信息必须位于机器人的规划帧中（即将更新）。(这一点即将更新。）
- `geometry_msgs::msg::PoseStamped` 位于 `pose_command_in_topic` 参数指定的主题上。

`Twist` 和 `Pose` 命令要求始终指定 `header.frame_id`。伺服节点（ROS 接口）的输出可以是 `trajectory_msgs::msg::JointTrajectory` 或 `std_msgs::msg::Float64MultiArray`（使用 `command_out_type` 参数选择），并发布在 `command_out_topic` 参数指定的主题上。

命令类型可通过 `ServoCommandType` 服务选择，参见 [`ServoCommandType`](https://github.com/moveit/moveit_msgs/blob/ros2/srv/ServoCommandType.srv) 定义。

命令行：

```bash
ros2 service call /<node_name>/switch_command_type moveit_msgs/srv/ServoCommandType "{command_type: 1}"
```

代码

```cpp
switch_input_client = node->create_client<moveit_msgs::srv::ServoCommandType>("/<node_name>/switch_command_type");
auto request = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
request->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
if (switch_input_client->wait_for_service(std::chrono::seconds(1)))
{
  auto result = switch_input_client->async_send_request(request);
  if (result.get()->success)
  {
    RCLCPP_INFO_STREAM(node->get_logger(), "Switched to input type: Twist");
  }
  else
  {
    RCLCPP_WARN_STREAM(node->get_logger(), "Could not switch input to: Twist");
  }
}
```

同样，也可以使用 `std_msgs::srv::SetBool` 类型的暂停服务 `<node_name>/pause_servo` 来暂停伺服功能。

使用 ROS 接口时，舵机的状态可在主题 `/<node_name>/status` 中查看，参见 [`ServoStatus`](https://github.com/moveit/moveit_msgs/blob/ros2/msg/ServoStatus.msg)定义。

启动 ROS 接口演示：

```bash
ros2 launch moveit_servo demo_ros_api.launch.py
```

演示程序运行后，即可通过键盘遥控机器人。

启动键盘演示：

```bash
ros2 run moveit_servo servo_keyboard_input
```

本例是在使用伺服功能打开一扇门的情况下使用姿势指令的一个示例。

```bash
ros2 launch moveit2_tutorials pose_tracking_tutorial.launch.py
```

## 信号平滑

查看 `servo_parameters.yaml`，注意到参数 `smoothing_filter_plugin_name`。这有助于消除机器人关节指令中的任何不规则现象，例如指令之间的间隔时间不一致。这可以大大减少机器人执行器的磨损，事实上，除非传入的指令足够平滑，否则很多机器人都不会移动。这些都是插件，任何用户都可以编写自己的插件。

当前的选项有：

`online_signal_smoothing::ButterworthFilterPlugin`： 这是一个非常简单的低通滤波器，比移动平均法稍先进一些。
- 优点：计算效率高，不会在联合空间中超调指令。
- 缺点：可能略微偏离笛卡尔空间中的 "直线运动"。没有明确限制执行器的颠簸或加速度。

`online_signal_smoothing::AccelerationLimitedPlugin`： 这是一种基于优化的算法，可遵守机器人的加速度限制（如果可行）。更多信息，请访问 https://github.com/moveit/moveit2/pull/2651
- 优点： 在运动学上可行的情况下，保持所需的运动方向。可用于 "急转弯"。确保不违反机器人关节加速度限制。
- 缺点 没有明确限制执行器的抽动。如果传入指令不可行，仍可能偏离预定运动方向。可能会超调。


- `online_signal_smoothing::RuckigFilterPlugin`： 该插件使用著名的 `Ruckig` 库来确保机器人运动始终遵守关节和加速度限制。更多信息，请访问 https://github.com/moveit/moveit2/pull/2956
- 优点： 最平滑的选项。某些工业机器人必须使用。
- 缺点： 有时会偏离预期运动方向。例如，在急转弯处往往会出现漩涡运动。要防止漩涡运动，需要在 `MoveIt` 伺服系统之外，对传入的指令进行额外的逻辑处理。可能会超调。

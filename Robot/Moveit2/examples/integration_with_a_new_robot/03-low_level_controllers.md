###### datetime:2025/02/25 11:15

###### author:nzb

# [低级控制器](https://moveit.picknik.ai/main/doc/examples/controller_configuration/controller_configuration_tutorial.html)

MoveIt 通常将机械臂运动命令发布到 `JointTrajectoryController`。本教程假设使用 `MoveGroup` 控制机器人，而不是 `MoveItCpp` 或 `MoveIt Servo`。最小设置如下：

1. **YAML 配置文件**：建议命名为 `moveit_controllers.yaml`。它告诉 MoveIt 哪些控制器可用，哪些关节与每个控制器相关联，以及 MoveIt 控制器接口类型（`FollowJointTrajectory` 或 `GripperCommand`）。示例控制器[配置文件](https://github.com/moveit/moveit_resources/blob/ros2/panda_moveit_config/config/moveit_controllers.yaml)。
2. **启动文件**：此启动文件必须加载 `moveit_controllers.yaml` 文件，并指定 `moveit_simple_controller_manager/MoveItSimpleControllerManager`。加载这些 YAML 文件后，它们将作为参数传递给 Move Group 节点。[示例 Move Group 启动文件](https://github.com/moveit/moveit_resources/blob/ros2/panda_moveit_config/launch/demo.launch.py)。
3. **启动相应的 `ros2_control` `JointTrajectoryControllers`**：这与 MoveIt 2 生态系统分开。[示例](https://github.com/ros-controls/ros2_control_demos) `ros2_control` 启动文件。每个 `JointTrajectoryController` 提供一个动作接口。给定上述 YAML 文件，MoveIt 会自动连接到此动作接口。
4. **注意**：不一定要为您的机器人使用 `ros2_control`。您可以编写专有的动作接口。实际上，99% 的用户选择 `ros2_control`。

## MoveIt 控制器管理器

控制器管理器的基类称为 `MoveItControllerManager`（MICM）。MICM 的一个子类称为 `Ros2ControlManager`（R2CM），它是与 `ros2_control` 交互的最佳方式。R2CM 可以解析来自 MoveIt 的轨迹命令中的关节名称，并激活相应的控制器。例如，它可以自动在同时控制单个关节组中的两个机械臂和单个机械臂之间切换。要使用 R2CM，只需在启动文件中设置 `moveit_manage_controllers = true`。[示例 R2CM 启动文件](https://github.com/moveit/moveit_resources/blob/ros2/panda_moveit_config/launch/demo.launch.py)。

## MoveIt 控制器接口

上述文本描述了关节轨迹控制器动作接口的启动。此外，MoveIt 支持通过动作接口控制平行夹爪。本节描述这两个选项的参数。

### **FollowJointTrajectory 控制器接口**
- **参数如下：**
  - **name**：控制器的名称。（请参阅下面的调试信息以获取重要说明）。
  - **action_ns**：控制器的动作命名空间。（请参阅下面的调试信息以获取重要说明）。
  - **type**：使用的动作类型（此处为 `FollowJointTrajectory`）。
  - **default**：默认控制器是 MoveIt 选择用于与特定关节集通信的主要控制器。
  - **joints**：此接口处理的所有关节的名称。

### **GripperCommand 控制器接口**
- **参数如下：**
  - **name**：控制器的名称。（请参阅下面的调试信息以获取重要说明）。
  - **action_ns**：控制器的动作命名空间。（请参阅下面的调试信息以获取重要说明）。
  - **type**：使用的动作类型（此处为 `GripperCommand`）。
  - **default**：默认控制器是 MoveIt 选择用于与特定关节集通信的主要控制器。
  - **joints**：此接口处理的所有关节的名称。
  - **command_joint**：控制夹爪实际状态的单个关节。这是唯一发送到控制器的值。必须是上述关节之一。如果未指定，则使用 `joints` 中的第一个条目。
  - **parallel**：设置此参数时，`joints` 的大小应为 2，命令将是两个关节的总和。

## 可选的允许轨迹执行持续时间参数

（TODO：更新为 ROS2）

对于每个控制器，可以设置 `allowed_execution_duration_scaling` 和 `allowed_goal_duration_margin` 参数。这些是控制器特定的全局参数 `trajectory_execution/allowed_execution_duration_scaling` 和 `trajectory_execution/allowed_goal_duration_margin` 的覆盖。与全局参数不同，控制器特定的参数不能在运行时动态重新配置。这些参数用于通过缩放预期执行持续时间并添加余量来计算允许的轨迹执行持续时间。如果超过此持续时间，轨迹将被取消。控制器特定的参数可以如下设置：

```yaml
controller_list:
- name: arm_controller
  action_ns: follow_joint_trajectory
  type: FollowJointTrajectory
  allowed_execution_duration_scaling: 1.2
  allowed_goal_duration_margin: 0.5
```

## 调试信息

（TODO：更新为 ROS2）

机器人上的 `FollowJointTrajectory` 或 `GripperCommand` 接口必须在命名空间 `/name/action_ns` 中进行通信。在上面的示例中，您应该能够在机器人上看到以下主题（使用 `ros2 topic list`）：

- `/panda_arm_controller/follow_joint_trajectory/goal`
- `/panda_arm_controller/follow_joint_trajectory/feedback`
- `/panda_arm_controller/follow_joint_trajectory/result`
- `/hand_controller/gripper_action/goal`
- `/hand_controller/gripper_action/feedback`
- `/hand_controller/gripper_action/result`

您还应该能够看到（使用 `ros2 topic info topic_name`）这些主题由机器人上的控制器和 `move_group` 节点发布/订阅。

## 重映射 `/joint_states` 主题

当您运行 [`move_group`](https://moveit.picknik.ai/main/doc/examples/move_group_interface/move_group_interface_tutorial.html) 节点时，可能需要将主题 `/joint_states` 重映射到 `/robot/joint_states`，否则 MoveIt 将无法获得关节的反馈。要进行此重映射，您可以为您的节点创建一个简单的启动文件，如下所示：

```python
# ros2 run 包名 节点名 --ros-args --remap 原话题名称:=新话题名称
ros2 run turtlesim turtlesim_node --ros-args --remap /turtle1/cmd_vel:=/cmd_vel

# luanch 文件
def generate_launch_description():
 
    return LaunchDescription([
        Node(package="turtlesim",executable="turtlesim_node",namespace="t1"),
        Node(package="turtlesim",
            executable="turtlesim_node",
            remappings=[("/turtle1/cmd_vel","/cmd_vel")]
        )
```

或者，您可以创建一个具有正确主题名称的订阅者，然后确保 `move_group` 的起始机器人状态对应于正确的关节角度，方法是使用此订阅者的回调。

## 轨迹执行管理器选项

有多种选项可以调整 `MoveIt` 中执行管道的行为和安全检查。在您的 `moveit_config` 包中编辑 `trajectory_execution.launch.xml` 文件以更改以下参数：

- **execution_duration_monitoring**：当为 `false` 时，如果轨迹在低级控制器侧完成时间超过预期，则不会抛出错误。
- **allowed_goal_duration_margin**：在触发轨迹取消之前允许超过预期执行时间（在缩放后应用）。
- **allowed_start_tolerance**：用于验证轨迹的第一个点与当前机器人状态匹配的允许关节值容差。如果设置为零，则在执行后跳过等待机器人停止。

## 示例控制器管理器

MoveIt 控制器管理器（某种程度上是一个误称）是您自定义低级控制器的接口。更好的理解方式是它们是控制器接口。对于大多数用例，如果您的机器人控制器已经为 `FollowJointTrajectory` 提供了 ROS 动作，则包含的 [`MoveItSimpleControllerManager`](https://github.com/moveit/moveit2/blob/main/moveit_plugins/moveit_simple_controller_manager) 是足够的。如果您使用 `ros_control`，则包含的 [`MoveItRosControllerface`](https://github.com/moveit/moveit2/blob/main/moveit_plugins/moveit_ros_control_interface) 也是理想的选择。

然而，对于某些应用，您可能需要更自定义的控制器管理器。此处提供了一个用于启动自定义控制器管理器的[示例模板](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/controller_configuration/src/moveit_controller_manager_example.cpp)。

## 仿真

如果您没有物理机器人，`ros2_control` 可以轻松模拟一个。不需要 `Ignition` 或 `Gazebo`；`RViz` 就足够了。`ros2_control_demos` 仓库中的所有[示例](https://github.com/ros-controls/ros2_control_demos)都是仿真的。

## 控制器切换和命名空间

（TODO：更新为 ROS2）

所有控制器名称的前缀都是其 `ros_control` 节点的命名空间。 因此，控制器名称不应包含斜线，也不能以 `/` 命名。对于特定节点，MoveIt 可以决定启动或停止哪些控制器。 由于 MoveIt 只处理已注册分配器插件的控制器名称，因此如果即将启动的控制器需要任何资源，MoveIt 会根据控制器声称的资源来停止控制器。

## 多节点控制器

`Ros2ControlManager` 有一个变体，称为 `Ros2ControlMultiManager`。`Ros2ControlMultiManager` 可用于多个 `ros_control` 节点。它通过为每个节点创建多个 `Ros2ControlManager` 来工作。它使用各自的命名空间实例化它们，并负责适当的委托。要使用它，必须将其添加到启动文件中。

```xml
<param name="moveit_controller_manager"
value="moveit_ros_control_interface::Ros2ControlMultiManager" />
```
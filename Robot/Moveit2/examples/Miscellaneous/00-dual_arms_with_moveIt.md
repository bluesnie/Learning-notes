###### datetime:2025/03/27 15:41

###### author:nzb

# [带 MoveIt 功能的双臂](https://moveit.picknik.ai/main/doc/examples/dual_arms/dual_arms_tutorial.html)

使用 `MoveIt` 控制两个或更多机械手需要很多配置步骤。幸运的是，随着时间的推移，这些步骤变得越来越简单。在此，我们提供了一个示例，并列出了从 1 个机械手到 X 个机械手所需的所有更改。

本示例中的启动和配置文件可在[此处获取](https://github.com/moveit/moveit_resources/tree/ros2/dual_arm_panda_moveit_config)。

## demo 

```shell
ros2 launch dual_arm_panda_moveit_config demo.launch.py
```

您会看到 `RViz` 启动了双臂系统。在下拉菜单中，你可以选择 `left_panda_arm` 或 `right_panda_arm` ，并可以使用其中任何一个来计划和执行运动。

![](../../imgs/rviz_dual_arms.png)

## 配置需要做哪些更改

- 为 `Panda` 臂 `xacro` 添加了前缀参数 `panda_arm_macro.urdf.xacro`。现在所有连杆和关节都以 `left_` 或 `right_` 为前缀。
- 添加 `left_initial_positions.yaml` 和 `right_initial_positions.yaml`。（这仅在模拟时才需要-使用硬件机器人时不适用。）将 `left_` 或 `right_` 前缀传递给 `panda.ros2_control.xacro` 以选择此文件。
- 确保 `panda.ros2_control.xacro` 中的所有关节都以前缀参数为前缀，这样它们对于左臂和右臂来说都是唯一的。
- 确保 `ros2_control` 宏的名称也以前缀为前缀，这样它才是唯一的：

```xml
<ros2_control name="${prefix}${name}" type="system">
```

- 在 `ros2_control` 配置文件 `ros2_controllers.yaml` 中枚举两个手臂所需的控制器。确保这些控制器从 `demo.launch.py`​​ 启动。
- 在 `panda.srdf` 中为每个手臂定义关节组。这会告诉 `MoveIt` 哪些关节构成每个手臂。关节组名为 `left_panda_arm` 和 `right_panda_arm` 。还要为每个手臂定义一个末端执行器。
- 在 `kinematics.yaml` 中为每个手臂定义一个运动学解算器。
- 在 `moveit_controllers.yaml` 中定义 `MoveIt` 可以执行轨迹的控制器。这里我们为每个手臂都有一个轨迹控制器。
- 同样在 `moveit_controllers.yaml` 中，定义 `MoveIt` 将使用的控制器管理策略。从配置角度来看，最简单的选项是 `moveit_ros_control_interface/Ros2ControlManager`。您还可以使用 `moveit_simple_controller_manager/MoveItSimpleControllerManager`，尽管它需要额外的命名空间和额外的关节枚举。
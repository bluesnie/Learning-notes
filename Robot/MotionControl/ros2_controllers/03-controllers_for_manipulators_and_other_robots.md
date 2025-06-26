###### datetime:2025/06/24 11:21

###### author:nzb

# 操作臂和其他机器人的控制器

## [Admittance Controller](https://control.ros.org/humble/doc/ros2_controllers/admittance_controller/doc/userdoc.html#admittance-controller)

`admittance` 控制器允许您从末端执行器（`TCP`）上测量的力实现零力控制。该控制器实现了 `ChainedControllerInterface` ，因此可以在其前面添加其他控制器，例如 `JointTrajectoryController` 。

该控制器需要一个外部运动学插件才能运行。运动学接口（`kinematics_interface`）存储库提供了 `admittance` 控制器可以使用的一个接口和实现。

- `ROS2` 控制器的接口
  - 参数：[参考链接](https://control.ros.org/humble/doc/ros2_controllers/admittance_controller/doc/userdoc.html#parameters)
  - 话题：
    - `~/joint_references (input topic) [trajectory_msgs::msg::JointTrajectoryPoint]`：控制器不在链式模式下时的目标关节指令。
    - `~/state (output topic) [control_msgs::msg::AdmittanceControllerState]`：发布内部状态的主题。
- `ros2_control` 接口
  - 参考：控制器导出了以 `<controller_name>/<joint_name>/[position|velocity]` 格式表示的 `position` 和 `velocity` 参考接口
  - 状态 
    - 状态接口使用 `joints` 和 `state_interfaces` 参数定义如下： `<joint>/<state_interface>` 。支持的状态接口为 `position` 、 `velocity` 和 `acceleration`，这些接口在 [`hardware_interface/hardware_interface_type_values.hpp`](https://github.com/ros-controls/ros2_control/blob/humble/hardware_interface/include/hardware_interface/types/hardware_interface_type_values.hpp) 中定义。如果某些接口未提供，将使用最后命令的接口进行计算。
    - 用于处理 `TCP` 扭矩 *力矩传感器* 语义组件（来自 [`*controller_interface*`](https://github.com/ros-controls/ros2_control/blob/humble/controller_interface/include/semantic_components/force_torque_sensor.hpp) 包）。接口具有 `ft_sensor.name` 前缀，构建的接口为： `<sensor_name>/[force.x|force.y|force.z|torque.x|torque.y|torque.z]` 。
  - 指令
    - 命令接口使用 `joints` 和 `command_interfaces` 参数定义如下： `<joint>/<command_interface>` 。支持的状态接口为 `position` 、 `velocity` 和 `acceleration` ，这些接口在 [`hardware_interface/hardware_interface_type_values.hpp`](https://github.com/ros-controls/ros2_control/blob/humble/hardware_interface/include/hardware_interface/types/hardware_interface_type_values.hpp) 中定义。

## [前向指令控制器(forward_command_controller)](https://control.ros.org/humble/doc/ros2_controllers/forward_command_controller/doc/userdoc.html#forward-command-controller)

- 硬件接口类型：该控制器可用于所有类型的命令接口。
- `ROS2` 控制器的接口
  - 话题：`~/commands (input topic) [std_msgs::msg::Float64MultiArray]`：目标关节指令
  - 参数：这个控制器使用 `generate_parameter_library` 来处理它的参数。

这是一个实现前馈控制器的基类。这个基类的具体实现可以在以下位置找到：

### [位置控制器(position_controllers)](https://control.ros.org/humble/doc/ros2_controllers/position_controllers/doc/userdoc.html#position-controllers)

这是一个使用“位置”关节命令接口工作的控制器集合，但在控制器级别可能接受不同的关节级命令，例如控制某个关节的位置以实现设定的速度。该软件包包含以下控制器：

- `position_controllers/JointGroupPositionController`
  - `ROS2` 控制器的接口
    - 话题：`~/commands (input topic) [std_msgs::msg::Float64MultiArray]`：关节位置命令
    - 参数：该控制器覆盖了 `forward_command_controller` 的接口参数， `joints` 参数是唯一必需的。

### [速度控制器(velocity_controllers)](https://control.ros.org/humble/doc/ros2_controllers/velocity_controllers/doc/userdoc.html#velocity-controllers)

这是一个使用“速度”关节命令接口工作的控制器集合，但在控制器级别可能接受不同的关节级命令，例如控制某个关节的速度以实现设定的位置。该软件包包含以下控制器：

- `velocity_controllers/JointGroupVelocityController`：这是使用“速度”关节接口工作的 `forward_command_controller` 的特化版本。
  - `ROS2` 控制器的接口
    - 话题：`~/commands (input topic) [std_msgs::msg::Float64MultiArray]`：关节的速度指令
    - 参数：该控制器覆盖了 `forward_command_controller` 的接口参数， `joints` 参数是唯一必需的。

### [力控制器(effort_controllers)](https://control.ros.org/humble/doc/ros2_controllers/effort_controllers/doc/userdoc.html#effort-controllers)

这是一个使用`effort`关节指令接口工作的控制器集合，但在控制器级别可能会接受不同的关节级指令，例如控制某个关节的力以达到设定的位置。该软件包包含以下控制器:

这是使用“`effort`”联合接口工作的[`forward_command_controller`](https://control.ros.org/humble/doc/ros2_controllers/forward_command_controller/doc/userdoc.html#forward-command-controller-userdoc)的专门化。
  - `ROS2` 控制器的接口
    - 话题：`~/commands (input topic) [std_msgs::msg::Float64MultiArray]`：关节的力矩指令
    - 参数：该控制器覆盖了 `forward_command_controller` 的接口参数， `joints` 参数是唯一必需的参数。

## [抓取器动作控制器(Gripper Action Controller)](https://control.ros.org/humble/doc/ros2_controllers/gripper_controllers/doc/userdoc.html#gripper-action-controller)

用于执行简单单自由度夹爪命令动作的控制器：

- `position_controllers/GripperActionController`
- `effort_controllers/GripperActionController`

- 参数：这些控制器使用 [`generate_parameter_library`](https://github.com/PickNikRobotics/generate_parameter_library) 来处理其参数。位于 [`src`](https://github.com/ros-controls/ros2_controllers/blob/humble/gripper_controllers/src/gripper_action_controller_parameters.yaml) 文件夹中的参数定义文件包含了控制器使用的所有参数的描述。


## [关节轨迹控制器(joint_trajectory_controller)](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html#joint-trajectory-controller)

用于在关节空间中执行一组关节轨迹的控制器。该控制器在时间上对点进行插值，以便它们的距离可以是任意的。即使只有一个点的轨迹也是接受的。轨迹被指定为一组在特定时间瞬间要达到的航点，控制器试图根据机械结构的限制尽可能好地执行这些航点。航点由位置组成，并且可选地包含速度和加速度。

### 硬件接口类型

目前，具有硬件接口类型 `position` 、 `velocity` 、 `acceleration` 和 `effort` （在此[定义](https://github.com/ros-controls/ros2_control/blob/humble/hardware_interface/include/hardware_interface/types/hardware_interface_type_values.hpp)）的关节作为命令接口支持以下组合：

- `position`
- `position`, `velocity`
- `position`, `velocity`, `acceleration`
- `velocity`
- `effort`

这意味着关节可以有一个或多个命令接口，同时应用以下控制律：

- 对于命令接口 `position` ，期望位置直接转发到关节，
- 对于命令接口 `acceleration` ，期望加速度直接转发到关节。
- 对于 `velocity` ( `effort` ) 命令接口，如果已配置（有关[参数](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/parameters.html#parameters)的详细信息），位置+速度轨迹跟踪误差将通过 `PID` 回路映射到 `velocity` ( `effort` ) 命令。

这就产生了以下允许的命令和状态接口组合：

- 使用命令接口 `position` ，状态接口没有限制。
- 使用命令接口 `velocity`
  - 如果命令接口 `velocity` 是唯一的，状态接口必须包含 `position`, `velocity` 。 
- 使用命令接口 `effort` ，状态接口必须包含 `position`, `velocity` 
- 使用命令接口 `acceleration` ，状态接口必须包含 `position`, `velocity` 。

状态接口存在进一步的限制：

- 如果缺少 `position` 接口，`velocity` 状态接口则无法使用。
- 如果缺少 `position` 和 `velocity` 接口，`acceleration` 状态接口则无法使用。”

### 其他特性

- 实时安全实现。
- 正确处理缠绕（连续）关节。
- 对系统时钟变化的鲁棒性：不连续的系统时钟变化不会导致已排队轨迹段的执行出现不连续性。

### 使用关节轨迹控制器(s) 

该控制器期望从硬件至少获得位置反馈。关节速度和加速度是可选的。目前该控制器内部不会从加速度中积分速度，也不会从速度中积分位置。因此，如果硬件仅提供加速度或速度状态，则必须在速度和位置硬件接口实现中进行积分，才能使用这些控制器。

关节轨迹控制器的通用版本在此包中实现。使用它的 `yaml` 文件可以是：

```yaml
controller_manager:
  ros__parameters:
    joint_trajectory_controller:
    type: "joint_trajectory_controller/JointTrajectoryController"

joint_trajectory_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
      - joint4
      - joint5
      - joint6

    command_interfaces:
      - position

    state_interfaces:
      - position
      - velocity

    state_publish_rate: 50.0
    action_monitor_rate: 20.0

    allow_partial_joints_goal: false
    open_loop_control: true
    constraints:
      stopped_velocity_tolerance: 0.01
      goal_time: 0.0
      joint1:
        trajectory: 0.05
        goal: 0.03
```

#### 抢占策略

- 任何时刻只能激活一个`action`目标，如果使用话题，则可以不激活任何目标。路径和目标容差仅针对激活目标的轨迹段进行检查。

- 当来自`action`接口的另一个命令抢占活动行动目标时，该目标将被取消，并通知客户端。轨迹将按定义的方式被替换，参见[轨迹替换](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html#joint-trajectory-controller-trajectory-replacement)。

- 从话题（而不是`action`接口）发送空的轨迹消息将覆盖当前`action`目标，并且不会中止动作。

#### 控制器接口的描述

- 参考：控制器尚未实现为可链式控制器
- 状态：状态接口使用 `joints` 和 `state_interfaces` 参数定义如下： `<joint>/<state_interface>`，合法的状态接口组合在上面的“硬件接口类型”章节中给出。
- 指令：发送轨迹到控制器的机制有两种：`action`和`topic`，详情见下文

两者都使用 `trajectory_msgs/msg/JointTrajectory` 消息来指定轨迹，并且如果 `allow_partial_joints_goal` 未设置为 `True` ，则需要为所有控制器关节指定值（而不是仅指定子集）。有关消息格式的更多信息，请参阅[轨迹表示](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html#joint-trajectory-controller-trajectory-representation)。

- **Actions**
  - `<controller_name>/follow_joint_trajectory [control_msgs::action::FollowJointTrajectory]`：用于向控制器发出指令的动作服务器，发送轨迹的主要方式是通过动作接口，当需要执行监控时应优先使用。
    - 动作目标允许指定要执行的轨迹，还可以（可选地）指定路径和目标容差。详情请参阅 [`JointTolerance`](https://github.com/ros-controls/control_msgs/blob/master/control_msgs/msg/JointTolerance.msg) 消息：
    - 当未指定容差时，将使用参数界面中给出的默认值（[参见参数详情](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/parameters.html#parameters)）。如果在轨迹执行过程中违反了容差，则动作目标将被中止，客户端将收到通知，并且当前位置将被保持。
    - 动作服务器在目标在指定公差范围内达到后，向客户端返回成功并继续执行最后一个指令点。
- **Subscriber**
  - `<controller_name>/joint_trajectory [trajectory_msgs::msg::JointTrajectory]`：用于向控制器发送指令的主题，主题接口是一种“发射即忘”的替代方案。如果你不关心执行监控，可以使用这个接口。在这种情况下，目标容差规范不会被使用，因为没有通知发送者的机制来处理容差违规。如果状态容差被违反，轨迹将被中止，当前位置将被保持。请注意，尽管可以通过 `~/query_state` 服务和 `~/state` 主题提供一定程度的监控，但与动作接口相比，实现起来要复杂得多。
- **Publishers**：`<controller_name>/controller_state [control_msgs::msg::JointTrajectoryControllerState]`：以控制器管理器的更新频率发布主题内部状态
- **Services**：`<controller_name>/query_state [control_msgs::srv::QueryTrajectoryState]`：查询任何未来时间的控制器状态
























## [PID 控制器(PID Controller)](https://control.ros.org/humble/doc/ros2_controllers/pid_controller/doc/userdoc.html#pid-controller)

使用 [`control_toolbox`](https://github.com/ros-controls/control_toolbox/) 包中的 `PidROS` 实现的 `PID` 控制器。该控制器可以通过主题发送参考量直接使用，或在具有前置或后置控制器的链中使用。它还支持使用参考量的导数及其反馈来实现二阶 `PID` 控制。

根据硬件的参考量/状态和命令接口，应使用 `PidROS` 的不同参数设置，例如：

- `reference/state POSITION; command VELOCITY –> PI CONTROLLER`
- `reference/state VELOCITY; command ACCELERATION –> PI CONTROLLER`
- `reference/state VELOCITY; command POSITION –> PD CONTROLLER`
- `reference/state ACCELERATION; command VELOCITY –> PD CONTROLLER`
- `reference/state POSITION; command POSITION –> PID CONTROLLER`
- `reference/state VELOCITY; command VELOCITY –> PID CONTROLLER`
- `reference/state ACCELERATION; command ACCELERATION –> PID CONTROLLER`
- `reference/state EFFORT; command EFFORT –> PID CONTROLLER`

> 理论上，我们可以滥用联合轨迹控制器（`JTC`），只向其发送一个参考点，以达到同样的目的。然而，我们并不建议这样做。如果需要在轨迹点之间使用线性、三次或五次插值，则应使用 `JTC`。`PID` 控制器不具备这种功能。`JTC` 中的 `PID` 术语具有不同的用途--它只能对硬件的速度或力度接口发出指令。

### 控制器的执行逻辑

控制器也可以在“前馈”模式下使用，其中前馈增益用于增加控制器的动态特性。如果参考和状态接口中只使用一种类型，则仅使用即时误差。如果使用两种类型，则第二个接口类型被视为第一种类型的导数。例如，一个有效的组合可以是 `position` 和 `velocity` 接口类型。

### 使用控制器

- `Pluginlib-Library`: `pid_controller` 
- `Plugin name`: `pid_controller/PidController`

### 控制器接口的描述

- 参考（来自先前的控制器）
  - `<reference_and_state_dof_names[i]>/<reference_and_state_interfaces[j]> [double]`
  - **注意**： `reference_and_state_dof_names[i]` 可以来自 `reference_and_state_dof_names` 参数，或者如果它是空的，则来自 `dof_names` 。
- 命令：`<dof_names[i]>/<command_interface> [double]`
- 状态
  - `<reference_and_state_dof_names[i]>/<reference_and_state_interfaces[j]> [double]`
  - **注意**： `reference_and_state_dof_names[i]` 可以来自 `reference_and_state_dof_names` 参数，或者如果它是空的，则来自 `dof_names` 。
- `Subscribers`
  - 如果控制器不在链式模式（ `in_chained_mode == false` )：`<controller_name>/reference [control_msgs/msg/MultiDOFCommand]`
  - 如果控制器参数 `use_external_measured_states == true`：`<controller_name>/measured_state [control_msgs/msg/MultiDOFCommand]`
- `Services`：`<controller_name>/set_feedforward_control [std_srvs/srv/SetBool]`
- `Publishers`：`<controller_name>/controller_state [control_msgs/msg/MultiDOFStateStamped]`
- [参数](https://control.ros.org/humble/doc/ros2_controllers/pid_controller/doc/userdoc.html#parameters)：`PID` 控制器使用 [`generate_parameter_library`](https://github.com/PickNikRobotics/generate_parameter_library) 来处理其参数。

## [gpio_controllers](https://control.ros.org/humble/doc/ros2_controllers/gpio_controllers/doc/userdoc.html#gpio-controllers)

这是一个用于 `GPIO`（`URDF` 中的 `<gpio>` 标签）硬件接口的控制器集合。

### gpio_command_controller

`gpio_command_controller` 允许用户暴露指定 `GPIO` 接口的命令接口，并发布配置的命令接口的状态接口。

- 控制器接口的描述
  - `/<controller_name>/gpio_states [ control_msgs/msg/DynamicJointState ]`: 发布给定 `GPIO` 接口的所有状态接口。
  - `/<controller_name>/commands [ control_msgs/msg/DynamicJointState ]`: 一个用于配置命令接口的订阅者。
- [参数](https://control.ros.org/humble/doc/ros2_controllers/gpio_controllers/doc/userdoc.html#parameters)

###### datetime:2025/06/23 11:50

###### author:nzb

# 轮式移动机器人的控制器

## [差分驱动移动机器人的控制器（diff_drive_controller）](https://control.ros.org/humble/doc/ros2_controllers/diff_drive_controller/doc/userdoc.html#diff-drive-controller)

- 它接收机器人身体的速度指令，并将这些指令转换为差分驱动底盘的轮子指令。
- 从硬件反馈计算里程计并发布。
- 对于移动机器人运动学的介绍以及此处使用的术语，请参阅[《轮式移动机器人运动学》](./00-wheeled_mobile_robot_kinematics.md)。

- 其他特性
  - 实时安全实现。
  - 里程计发布
  - 任务空间速度、加速度和加加速度限制
  - 命令超时后自动停止

- 控制器接口的描述
  - 反馈：作为反馈接口类型，使用关节的位置（ `hardware_interface::HW_IF_POSITION` ）或速度（ `hardware_interface::HW_IF_VELOCITY` ，如果参数 `position_feedback=false` 设置）。
  - 输出：关节速度（ `hardware_interface::HW_IF_VELOCITY` ）被使用。

- `ROS2` 接口
    - 订阅者
      - `~/cmd_vel [geometry_msgs/msg/TwistStamped]`：控制器速度指令，如果 `use_stamped_vel=true` 。控制器提取线性速度的 `x` 分量和角速度的 `z` 分量。其他分量的速度被忽略。
      - `~/cmd_vel_unstamped [geometry_msgs::msg::Twist]`：控制器速度指令，如果 `use_stamped_vel=false` 。控制器提取线性速度的 `x` 分量和角速度的 `z` 分量。其他分量的速度被忽略。
    - 发布者
      - `~/odom [nav_msgs::msg::Odometry]`：这表示机器人在自由空间中的位置和速度的估计值。
      - `/tf [tf2_msgs::msg::TFMessage]`：`tf` 树。仅当 `enable_odom_tf=true` 存在时发布。
      - `~/cmd_vel_out [geometry_msgs/msg/TwistStamped]`：控制器使用的速度指令，其中已应用限制。仅当 `publish_limited_velocity=true` 时发布。
    - 参数：[参考链接](https://control.ros.org/humble/doc/ros2_controllers/diff_drive_controller/doc/userdoc.html#parameters)

## [万向轮驱动控制器(mecanum_drive_controller)](https://control.ros.org/humble/doc/ros2_controllers/mecanum_drive_controller/doc/userdoc.html#mecanum-drive-controller)

用于配备 `mecanum` 驱动（四个 `mecanum` 轮）的移动机器人控制器的共享功能库。该库实现了通用的里程计和更新方法，并定义了主要接口。

- 控制器的执行逻辑 

    控制器使用速度输入，即带有时间戳的 `Twist` 消息，其中使用线性组件 `x` 、 `y` 和角组件 `z` 。其他组件的值被忽略。在链模式中，控制器提供三个参考接口，一个用于线速度，一个用于转向角位置。其他相关特性包括：

    - 以 `Odometry` 和 `TF` 消息形式发布里程计
    - 基于参数输入命令超时。

关于里程计计算：在 `DiffDRiveController` 中，速度会被过滤掉，但我们倾向于返回原始数据并让用户自行进行后处理。我们更倾向于这种方式，因为过滤会引入延迟（这使得解释和比较行为曲线变得困难）。

- 控制器接口的描述
  - 参考（来自前一个控制器）
    - 当控制器处于串联模式时，它会暴露以下参考，这些参考可以被前一个控制器命令：
    - `<controller_name>/linear/x/velocity`, `in m/s`
    - `<controller_name>/linear/y/velocity`, `in m/s`
    - `<controller_name>/linear/z/velocity`, `in rad/s`
  - 指令
    - `<*_wheel_command_joint_name>/velocity`, `in rad/s`
  - 状态
    - `<joint_name>/velocity`, `in rad/s` 

  - 订阅者
    - 当控制器不在串联模式时使用（ `in_chained_mode == false` ）。
    - `<controller_name>/reference [ geometry_msgs/msg/TwistStamped ]` 控制器的速度指令，如果 `use_stamped_vel == true` 。
    - `<controller_name>/reference_unstamped [ geometry_msgs/msg/Twist ]` 控制器的速度指令，如果 `use_stamped_vel == false` 。
  - 发布者
    - `<controller_name>/odometry [nav_msgs/msg/Odometry]`
    - `<controller_name>/tf_odometry [tf2_msgs/msg/TFMessage]`
    - `<controller_name>/controller_state [control_msgs/msg/MecanumDriveControllerState]`
  - 参数：[参考链接](https://control.ros.org/humble/doc/ros2_controllers/mecanum_drive_controller/doc/userdoc.html#parameters)

## [转向控制器库(steering_controllers_library)](https://control.ros.org/humble/doc/ros2_controllers/steering_controllers_library/doc/userdoc.html#steering-controllers-library)

- 一个用于具有转向驱动（2 个自由度）的移动机器人控制器的库，具有所谓的非完整约束。
- 该库实现了通用的里程计和更新方法，并定义了主要接口。
- 更新方法仅使用逆运动学，它没有实现任何反馈控制回路，例如路径跟踪控制器等。
- 对于移动机器人运动学的介绍以及此处使用的术语，请参阅《轮式移动机器人运动学》。

### 控制器的执行逻辑

控制器使用速度输入，即带时间戳或不带时间戳的 `twist` 消息，其中使用线性 `x` 和角速度 `z` 分量。其他分量中的值被忽略。
在链模式中，控制器提供两个参考接口，一个用于线速度，一个用于转向角位置。其他相关功能包括：

- 支持前轮和后轮转向配置；
- 以 `Odometry` 和 `TF` 消息形式发布里程计
- 基于参数输入命令超时。

轮子的指令通过 `odometry` 库计算得出，该库基于具体的运动学模型计算牵引和转向指令。

### 当前实现的运动学模型

#### [自行车(bicycle_steering_controller)](https://control.ros.org/humble/doc/ros2_controllers/bicycle_steering_controller/doc/userdoc.html#bicycle-steering-controller)：具有一个转向关节和一个驱动关节；

该控制器实现了具有两个轴和两个轮子的运动学，其中一个轴上的轮子是固定的（牵引/驱动），另一个轴上的轮子是可转向的。

控制器期望有一个用于牵引的指令关节和一个用于转向的指令关节。如果你的阿克曼转向车辆在轴上使用差速器，那么你应该使用这个控制器，因为你可以为轴线中间的虚拟轮命令一个牵引速度和一个转向角。

有关控制器执行和接口的更多详细信息，请查看[转向控制器库](https://control.ros.org/humble/doc/ros2_controllers/steering_controllers_library/doc/userdoc.html#steering-controllers-library-userdoc)。

- 参数：[参考链接](https://control.ros.org/humble/doc/ros2_controllers/bicycle_steering_controller/doc/userdoc.html#parameters)

#### [三轮车(tricycle_steering_controller)](https://control.ros.org/humble/doc/ros2_controllers/tricycle_steering_controller/doc/userdoc.html#tricycle-steering-controller)：具有一个转向关节和两个驱动关节；

该控制器使用两个轴和三个轮实现运动学，其中每个轴上有两个轮是固定的（牵引/驱动），另一个轴上的轮是可转向的。

控制器期望有两个牵引关节，每个固定轮一个，还有一个转向关节。

有关控制器执行和接口的更多详细信息，请查看[转向控制器库](https://control.ros.org/humble/doc/ros2_controllers/steering_controllers_library/doc/userdoc.html#steering-controllers-library-userdoc)。


#### [阿克曼(ackermann_steering_controller)](https://control.ros.org/humble/doc/ros2_controllers/ackermann_steering_controller/doc/userdoc.html#ackermann-steering-controller)：具有两个转向和两个驱动关节。

该控制器实现了具有两个轴和四个轮子的运动学，其中一个轴上的轮子是固定的（牵引/驱动）轮，另一个轴上的轮子是可转向的。

该控制器期望有两个用于牵引的指令关节，每个固定轮一个，以及两个用于转向的指令关节，每个轮子一个。

有关控制器执行和接口的更多详细信息，请查看[转向控制器库](https://control.ros.org/humble/doc/ros2_controllers/steering_controllers_library/doc/userdoc.html#steering-controllers-library-userdoc)。


### 控制器接口的描述

- 参考（来自前一个控制器）
  - 当控制器处于链式模式（ `in_chained_mode == true` ）时使用。
    - `<controller_name>/linear/velocity` `double, in m/s`
    - `<controller_name>/angular/velocity` `double, in rad/s`

- 命令接口
  - 如果参数 `front_steering == true`
    - `<front_wheels_names[i]>/position` `double, in rad`
    - `<rear_wheels_names[i]>/velocity` `double, in rad/s`
  - 如果参数 `front_steering == false`
    - `<front_wheels_names[i]>/velocity` `double, in rad/s`
    - `<rear_wheels_names[i]>/position` `double, in rad`

- 状态接口
  - 根据 `position_feedback` ，期望不同的反馈类型
    - `position_feedback == true –> TRACTION_FEEDBACK_TYPE = position`
    - `position_feedback == false –> TRACTION_FEEDBACK_TYPE = velocity`
  - 如果参数 `front_steering == true`
    - `<front_wheels_names[i]>/position` `double, in rad`
    - `<rear_wheels_names[i]>/<TRACTION_FEEDBACK_TYPE>` `double, in rad or rad/s`
  - 如果参数 `front_steering == false`
    - `<front_wheels_names[i]>/<TRACTION_FEEDBACK_TYPE>` `double, in rad or rad/s`
    - `<rear_wheels_names[i]>/position` `double, in rad`
- 订阅者
  - 当控制器不在链式模式下使用（ `in_chained_mode == false` ）。
    - `use_stamped_vel == true -> <controller_name>/reference [geometry_msgs/msg/TwistStamped]`
    - `use_stamped_vel == false -> <controller_name>/reference_unstamped [geometry_msgs/msg/Twist]`
- 发布者
  - `<controller_name>/odometry [nav_msgs/msg/Odometry]`
  - `<controller_name>/tf_odometry [tf2_msgs/msg/TFMessage]`
  - `<controller_name>/controller_state [control_msgs/msg/SteeringControllerStatus]`
- 参数：[参考链接](https://control.ros.org/humble/doc/ros2_controllers/steering_controllers_library/doc/userdoc.html#parameters)


## [三轮车控制器(tricycle_controller)](https://control.ros.org/humble/doc/ros2_controllers/tricycle_controller/doc/userdoc.html#tricycle-controller)

适用于单轮双驱动移动机器人的控制器，包括牵引力和转向功能。例如，前轮为驱动轮、后轴上安装两个从动轮的三轮机器人。

控制输入为机器人 `base_link` 的旋转指令，这些指令被转换为三轮驱动底盘的牵引力和转向指令。里程计根据硬件反馈计算并发布。

对于移动机器人运动学的介绍以及此处使用的术语，请参阅[《轮式移动机器人运动学》](https://control.ros.org/humble/doc/ros2_controllers/doc/mobile_robot_kinematics.html#mobile-robot-kinematics)。

- 其他特性：实时安全实现。里程计发布速度、加速度和加加速度限制。指令超时后自动停止
- ROS 2 接口
  - 订阅者
    - `~/cmd_vel [geometry_msgs/msg/TwistStamped]`

























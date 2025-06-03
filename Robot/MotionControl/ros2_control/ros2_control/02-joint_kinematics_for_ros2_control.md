###### datetime:2025/05/09 10:46

###### author:nzb

# [ros2_control 的关节运动学](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/joints_userdoc.html)

## 命名

- **Degrees of Freedom (DoF)**：自由度 （`DoF`）：在物理学中，机械系统的自由度 （`DoF`） 是定义其配置或状态的独立参数的数量。
- **Joint  关节**

关节是两个链接之间的连接。在 ROS 生态系统中，三种类型更典型：`Revolute`（具有位置限制的铰链关节）、`Continuous`（没有任何位置限制的连续铰链）或 `Prismatic`（沿轴移动的滑动关节）。

通常，关节可以是致动的或非致动的，也称为被动的。被动关节是没有自己的驱动机构，而是允许通过外力或被其他关节被动移动的关节。被动关节的 `DoF` 可以为 1，例如摆锤，也可以是 `DoF` 为零的并联运动机构的一部分。

- **Serial Kinematics  串行运动学**

串行运动学是指机器人机械手中关节的排列，其中每个关节都独立于其他关节，关节数量等于运动链的 `DoF`。

一个典型的例子是具有 6 个旋转关节的工业机器人，具有 6-DoF。每个关节都可以独立启动，并且末端执行器可以移动到工作区中的任何位置和方向。

- **Kinematic Loops  运动学循环**

另一方面，运动学回路，也称为闭环机构，涉及连接在运动链中并一起驱动的多个关节。这意味着关节是耦合的，不能独立移动：通常，DoF 的数量小于关节的数量。这种结构是并联运动机构的典型结构，其中末端执行器通过多个运动链连接到底座。

例如，四杆联动装置由四个连杆和四个铰接组成。尽管有四个关节，但它可以有一个或两个致动器，因此也可以有一个或两个 `DoF`。此外，我们可以说我们有一 （2） 个驱动关节和三 （2） 个被动关节，它们必须满足机构的运动学约束。

## URDF

`URDF` 是描述 `ROS` 中机器人运动学的默认格式。但是，仅支持串行运动链，除了所谓的模拟关节。有关更多详细信息，请参阅 `URDF` [规范](http://wiki.ros.org/urdf/XML/joint) 。

模拟关节可以在 `URDF` 中按以下方式定义

```xml
<joint name="right_finger_joint" type="prismatic">
  <axis xyz="0 1 0"/>
  <origin xyz="0.0 -0.48 1" rpy="0.0 0.0 0.0"/>
  <parent link="base"/>
  <child link="finger_right"/>
  <limit effort="1000.0" lower="0" upper="0.38" velocity="10"/>
</joint>
<joint name="left_finger_joint" type="prismatic">
  <mimic joint="right_finger_joint" multiplier="1" offset="0"/>
  <axis xyz="0 1 0"/>
  <origin xyz="0.0 0.48 1" rpy="0.0 0.0 3.1415926535"/>
  <parent link="base"/>
  <child link="finger_left"/>
  <limit effort="1000.0" lower="0" upper="0.38" velocity="10"/>
</joint>
```

模拟关节是现实世界的抽象。例如，它们可用于描述

- 简单的闭环运动学，关节位置和速度具有线性依赖性
- 与皮带连接的链节，如皮带和滑轮系统或伸缩臂
- 被动关节的简化模型，例如末端执行器处的钟摆始终朝下
- 抽象复杂的驱动关节组，其中多个关节由低级控制回路直接控制并同步移动。如果不给出实际示例，这可能是多个电机具有各自的电力电子元件，但使用相同的设定值进行命令。

`URDF` 中定义的模拟关节从资源管理器中解析，并存储在 `HardwareInfo` 类型的类变量中，该变量可由硬件组件访问。模拟关节不能有命令接口，但可以有状态接口。

```xml
<ros2_control>
  <joint name="right_finger_joint">
    <command_interface name="effort"/>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>
  <joint name="left_finger_joint">
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>
</ros2_control>
```

在正式发布的软件包中，以下软件包已经在使用此信息：

- [mock_components (generic system)](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/mock_components_userdoc.html#mock-components-userdoc)
- [gazebo_ros2_control](https://control.ros.org/humble/doc/gazebo_ros2_control/doc/index.html#gazebo-ros2-control)
- [ign_ros2_control](https://control.ros.org/humble/doc/gz_ros2_control/doc/index.html#ign-ros2-control)

由于 `URDF` 仅指定运动学，因此 `mimic` 标签必须独立于 `ros2_control` 中使用的硬件接口类型。这意味着我们按以下方式解释此信息：


- **position = multiplier * other_joint_position + offset**
- **velocity = multiplier * other_joint_velocity**
- 
如果有人出于任何原因想要在不更改 `URDF` 的情况下停用 `mimic` 关节行为，则可以通过在 `XML` 的 `<ros2_control>` 部分中设置关节标签的属性 `mimic=false` 来完成。

```xml
<joint name="left_finger_joint" mimic="false">
  <state_interface name="position"/>
  <state_interface name="velocity"/>
  <state_interface name="effort"/>
</joint>
```

## 传输接口

机械传动转换功/流量变量，使其乘积（功率）保持不变。线性和旋转域的力变量是力和扭矩;而流量变量分别为 `Linear Velocity` 和 `Angular Velocity`。

在机器人技术中，习惯上将传动装置放置在执行器和关节之间。此接口遵循此命名来标识转换的输入和输出空间。提供的接口允许在 `Actuator` 和关节空间之间进行 `Effort、Velocity` 和 `Position` 的双向映射。位置不是幂变量，但可以使用速度图加上表示致动器和关节零点之间偏移的积分常数来实现映射。

`transmission_interface` 为插件提供了一个基类和一些实现，这些插件可以通过自定义硬件组件进行集成和加载。它们不会由任何硬件组件或 `gazebo` 插件自动加载，每个硬件组件都负责加载适当的传输接口，以将执行器读数映射到联合读数。

目前有以下实现可用：

- `SimpleTransmission`：具有恒定减速比且无额外动态的简单传动。
- `DifferentialTransmission`：具有两个致动器和两个关节的差速变速器。
- `FourBarLinkageTransmission`：带有两个执行器和两个关节的四杆联动变速器。

有关更多信息，请参阅 [`example_8`](https://control.ros.org/humble/doc/ros2_control_demos/example_8/doc/userdoc.html#ros2-control-demos-example-8-userdoc) 或 [`transmission_interface`](http://docs.ros.org/en/humble/p/transmission_interface/index.html) 文档。

## 模拟闭环运动链

根据仿真插件的不同，可以使用不同的方法来仿真闭环运动链。以下列表概述了可用的模拟插件及其功能：

- `gazebo_ros2_control`
  - `mimic joints`：模拟关节
  - `URDF` 中的 `<gazebo>` 标签支持闭环运动学，例如，请参阅[此处](http://classic.gazebosim.org/tutorials?tut=kinematic_loop) 。
- `gz_ros2_control`
  - `mimic joints`：模拟关节
  - 目前尚不直接支持闭环运动学，但可以通过自定义插件使用 `DetachableJoint` 来实现。请关注此问题 ，了解有关[此主题](https://github.com/gazebosim/gz-physics/issues/25)的更新。

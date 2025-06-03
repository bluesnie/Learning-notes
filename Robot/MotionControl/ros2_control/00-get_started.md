###### datetime:2025/05/07 11:01

###### author:nzb

# [开始使用](https://control.ros.org/humble/doc/getting_started/getting_started.html)

##  [安装](https://control.ros.org/humble/doc/getting_started/getting_started.html#installation)

## 架构

`ros2_control` 框架的源代码可以在 [`ros2_control`](https://github.com/ros-controls/ros2_control) 和 [`ros2_controllers`](https://github.com/ros-controls/ros2_controllers) `GitHub` 存储库中找到。下图显示了 `ros2_control` 框架的体系结构。

![](../imgs/components_architecture.png)

### 控制器管理器

控制器管理器（CM） 连接 `ros2_control` 框架的控制器和硬件抽象端。它还通过 `ROS` 服务作为用户的入口点。`CM` 在没有执行程序的情况下实现节点，以便可以将其集成到自定义设置中。但是，通常建议使用 `controller_manager` 包中 `ros2_control_node` 文件中实现的默认 `node-setup`。本手册假定您使用此默认 `node-setup`。

一方面，`CM` 管理（例如加载、激活、停用、卸载）控制器及其所需的接口。另一方面，它可以（通过 `Resource Manager`）访问硬件组件，即它们的接口。`Controller Manager` 匹配必需的接口和提供的接口，在启用时授予控制器对硬件的访问权限，或者在存在访问冲突时报告错误。

控制循环的执行由 `CM` 的 `update（）` 方法管理。它从硬件组件中读取数据，更新所有活动控制器的输出，并将结果写入组件。

### 资源管理器

`Resource Manager （RM）` 抽象化了 `ros2_control` 框架的物理硬件及其驱动程序（称为硬件组件 ）。`RM` 使用 `pluginlib-library` 加载组件，管理其生命周期以及组件的状态和命令接口。`RM` 提供的抽象允许重用已实现的硬件组件，例如机器人和夹持器，无需任何实现，以及用于状态和命令接口的灵活硬件应用程序，例如，用于电机控制和编码器读取的单独硬件/通信库。

在控制循环执行中，`RM` 的 `read（）` 和 `write（）` 方法处理与硬件组件的通信。

### 控制器

`ros2_control` 框架中的控制器基于控制理论。他们将参考值与测量输出进行比较，并根据此误差计算系统的输入。控制器是从 `ControllerInterface`（`ros2_control` 中的 `controller_interface` 包）派生的对象，并使用 `pluginlib-library` 导出为插件。有关控制器的示例，请查看 `ros2_controllers` 存储库中的 [`ForwardCommandController`](https://github.com/ros-controls/ros2_controllers/blob/master/forward_command_controller/src/forward_command_controller.cpp) 实现 。控制器生命周期基于 `LifecycleNode` 类，该类实现 `Node` 生命周期设计文档中描述的状态机。

执行 `control-loop` 时，调用 `update（）` 方法。该方法可以访问最新的硬件状态，并使控制器能够写入硬件命令接口。

### 用户界面

用户使用 `Controller Manager` 的服务与 `ros2_control` 框架进行交互。有关服务及其定义的列表，请检查 `controller_manager_msgs` 包中的 `srv` 文件夹。

虽然服务调用可以直接从命令行或通过节点使用，但存在一个用户友好的命令行界面 （CLI），它与 `ros2 CLI` 集成。它支持自动完成，并提供了一系列常用命令。基本命令是 `ros2 control`。有关 `CLI` 功能的说明，[请参阅命令行界面 （CLI） 文档](https://control.ros.org/humble/doc/ros2_control/ros2controlcli/doc/userdoc.html#ros2controlcli-userdoc) 。

## 硬件组件

硬件组件实现与物理硬件的通信，并在 `ros2_control` 框架中表示其抽象。组件必须使用 `pluginlib-library` 导出为插件。`Resource Manager` 动态加载这些插件并管理其生命周期。

有三种基本类型的组件：

- System

复杂（多自由度）机器人硬件，例如工业机器人。与执行器组件`Actuator`的主要区别在于，它能够使用复杂的传动装置，例如人形机器人手所需的传动装置。该组件具有读写功能。当只有一个逻辑通信通道连接到硬件（例如 KUKA-RSI）时，可以使用它。

- Sensor

机器人硬件用于感应其环境。传感器组件与关节（例如编码器）或连杆（例如力扭矩传感器）相关。此组件类型仅具有读取功能。  

- Actuator

简单的 （1 自由度） 机器人硬件，如电机、阀门等。平动电机实现仅与一个关节相关。此组件类型具有读取和写入功能。如果不可能，则读取不是强制性的（例如，使用 `Arduino` 板进行直流电机控制）。如果其硬件支持模块化设计，例如与每个电机独立进行 `CAN` 通信，则执行器类型也可以与多自由度机器人一起使用。

硬件组件的详细说明在 [Hardware Access through Controllers](https://github.com/ros-controls/roadmap/blob/master/design_drafts/hardware_access.md) 设计文档中给出。

### 在 URDF 中的硬件描述 

`ros2_control` 框架使用机器人 `URDF` 文件中的 `<ros2_control>` 标签来描述其组件，即硬件设置。所选结构可以将多个 `xacro-macro` 一起跟踪到一个中，而无需进行任何更改。以下示例显示了一个位置控制机器人，该机器人具有 `2-DOF （RRBot）`、外部 `1-DOF` 力-扭矩传感器和一个外部控制的 `1-DOF` 平行夹具作为其末端执行器。有关更多示例和详细说明，请查看 [`ros2_control_demos`](https://control.ros.org/humble/doc/ros2_control_demos/doc/index.html#ros2-control-demos) 站点和 `ROS 2` 控制组件 `URDF` [示例设计文档](https://github.com/ros-controls/roadmap/blob/master/design_drafts/components_architecture_and_urdf_examples.md) 。

```xml
<ros2_control name="RRBotSystemPositionOnly" type="system">
 <hardware>
   <plugin>ros2_control_demo_hardware/RRBotSystemPositionOnlyHardware</plugin>
   <param name="example_param_write_for_sec">2</param>
   <param name="example_param_read_for_sec">2</param>
 </hardware>
 <joint name="joint1">
   <command_interface name="position">
     <param name="min">-1</param>
     <param name="max">1</param>
   </command_interface>
   <state_interface name="position"/>
 </joint>
 <joint name="joint2">
   <command_interface name="position">
     <param name="min">-1</param>
     <param name="max">1</param>
   </command_interface>
   <state_interface name="position"/>
 </joint>
</ros2_control>
<ros2_control name="RRBotForceTorqueSensor1D" type="sensor">
 <hardware>
   <plugin>ros2_control_demo_hardware/ForceTorqueSensor1DHardware</plugin>
   <param name="example_param_read_for_sec">0.43</param>
 </hardware>
 <sensor name="tcp_fts_sensor">
   <state_interface name="force"/>
   <param name="frame_id">rrbot_tcp</param>
   <param name="min_force">-100</param>
   <param name="max_force">100</param>
 </sensor>
</ros2_control>
<ros2_control name="RRBotGripper" type="actuator">
 <hardware>
   <plugin>ros2_control_demo_hardware/PositionActuatorHardware</plugin>
   <param name="example_param_write_for_sec">1.23</param>
   <param name="example_param_read_for_sec">3</param>
 </hardware>
 <joint name="gripper_joint ">
   <command_interface name="position">
     <param name="min">0</param>
     <param name="max">50</param>
   </command_interface>
   <state_interface name="position"/>
   <state_interface name="velocity"/>
 </joint>
</ros2_control>
```

### 迁移此框架到你机器人上

要运行 `ros2_control` 框架，请执行以下作。示例文件可以在 [`ros2_control_demos`](https://github.com/ros-controls/ros2_control_demos) 存储库中找到。

- 使用控制器管理器和两个控制器的配置创建一个 `YAML` 文件。（`RRBot` 的示例配置 ）
- 使用所需的 `<ros2_control>` 标记扩展机器人的 `URDF` 描述。建议使用宏文件 （`xacro`） 而不是纯 `URDF`。（ `RRBot` 的 `URDF` 示例 ）
- 创建启动文件以使用 `Controller Manager` 启动节点。您可以使用默认的 `ros2_control` 节点 （推荐）或将控制器管理器集成到您的软件堆栈中。（ `RRBot` 的示例启动文件 ）

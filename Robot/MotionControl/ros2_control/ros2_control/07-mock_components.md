###### datetime:2025/05/21 14:30

###### author:nzb

# [模拟组件](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/mock_components_userdoc.html)

模拟组件是硬件组件（即系统、传感器和执行器）的简单“模拟”。它们通过镜像命令到状态来提供理想的行为。相应的硬件接口可以代替真实硬件用于离线测试 `ros2_control` 框架。主要优势在于无需硬件即可测试框架内部的“管道”。这意味着你可以测试你的控制器、广播器、启动文件，甚至与 MoveIt 等集成。主要目的是减少物理硬件的调试时间，并提高你的开发效率。

## 通用系统

该组件实现了 `hardware_interface::SystemInterface` 支持命令和状态接口。有关硬件组件的更多信息，请查看[详细文档](https://control.ros.org/humble/doc/getting_started/getting_started.html#overview-hardware-components)。


功能：

- 支持 `mimic` 关节
- 将指令镜像到带和不带偏移的状态
- 用于从外部节点设置传感器数据的模拟命令接口（与转发[控制器](https://control.ros.org/humble/doc/ros2_controllers/forward_command_controller/doc/userdoc.html#forward-command-controller-userdoc)结合使用）

### 参数

一个包含所有可选参数（带默认值）的完整示例：

```xml
<ros2_control name="MockHardwareSystem" type="system">
  <hardware>
    <plugin>mock_components/GenericSystem</plugin>
    <param name="calculate_dynamics">false</param>
    <param name="custom_interface_with_following_offset"></param>
    <param name="disable_commands">false</param>
    <param name="mock_gpio_commands">false</param>
    <param name="mock_sensor_commands">false</param>
    <param name="position_state_following_offset">0.0</param>
  </hardware>
  <joint name="joint1">
    <command_interface name="position"/>
    <command_interface name="velocity"/>
    <state_interface name="position">
      <param name="initial_value">3.45</param>
    </state_interface>
    <state_interface name="velocity"/>
    <state_interface name="acceleration"/>
  </joint>
  <joint name="joint2">
    <command_interface name="velocity"/>
    <command_interface name="acceleration"/>
    <state_interface name="position">
      <param name="initial_value">2.78</param>
    </state_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="acceleration"/>
  </joint>
  <gpio name="flange_vacuum">
    <command_interface name="vacuum"/>
    <state_interface name="vacuum" data_type="double"/>
  </gpio>
</ros2_control>
```

请参考 [`example_2`](https://control.ros.org/humble/doc/ros2_control_demos/example_2/doc/userdoc.html#ros2-control-demos-example-2-userdoc)，了解如何使用 `calculate_dynamics` ，或者参考 `example_10`，了解如何与 `GPIO` 接口结合使用。

#### 组件参数

- `calculate_dynamics (optional; boolean; default: false)`：通过使用欧拉前向积分或有限差分来计算状态。
- `custom_interface_with_following_offset (optional; string; default: “”)`：将偏移后的命令映射到自定义接口。
- `disable_commands (optional; boolean; default: false)`：禁止将命令镜像到状态。此选项有助于在硬件没有故障时模拟与硬件的错误连接，或者突然没有来自硬件接口的反馈。或者它可以帮助你在硬件没有反馈的情况下测试你的设置，即，在开环配置中。
- `mock_gpio_commands (optional; boolean; default: false)`：创建用于通过外部命令模拟 `GPIO` 状态的假命令接口。这些接口通常由前向控制器使用，以从 `ROS` 世界提供访问。
- `mock_sensor_commands (optional; boolean; default: false)`：创建用于通过外部命令模拟传感器测量的假命令接口。这些接口通常由前向控制器使用，以从 `ROS` 世界提供访问。
- `position_state_following_offset (optional; double; default: 0.0)`：当镜像到状态时，命令值增加的跟随偏移量。仅当 `custom_interface_with_following_offset` 为 `false` 时应用。

### 每个关节参数

- `mimic (optional; string)`: 模仿（可选；字符串）;模拟关节的预定义名称。这通常与平行夹爪的概念一起使用。示例： `<param name="mimic">joint1</param>` 。
- `multiplier (optional; double; default: 1; used if mimic joint is defined)`乘数（可选；双精度浮点数；默认值：1；如果定义了模拟关节则使用）
    模拟关节（在 `mimic` 参数中定义）的值乘数。示例： `<param name="multiplier">-2</param> `

### 每个接口的参数 

- `initial_value (optional; double)`：启动后直接获取的某些状态接口的初始值。示例：

```xml
<state_interface name="position">
  <param name="initial_value">3.45</param>
</state_interface>
```

注意：此参数与 `gazebo` 和 `gazebo` 经典插件用于关节接口共享。对于 `Mock` 组件，也可以为 `gpio` 或传感器状态接口设置初始值。
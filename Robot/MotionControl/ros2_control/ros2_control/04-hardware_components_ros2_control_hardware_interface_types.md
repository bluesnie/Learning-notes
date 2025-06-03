###### datetime:2025/05/19 16:46

###### author:nzb

# [ros2_control 硬件接口类型](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/hardware_interface_types_userdoc.html)

`ros2_control` 框架提供了一套硬件接口类型，可用于为特定机器人或设备实现硬件组件。以下各节将描述不同的硬件接口类型及其使用方法。

## Joints

`<joint>`-标签组合了与物理机器人和执行器的关节相关的接口。它们具有命令和状态接口，用于设置硬件的目标值并读取其当前状态。通过 `joint_state_broadcaster`，关节的状态接口可以作为一个 `ROS` 主题发布

## Sensors

`<sensor>` -标签用于组合多个状态接口，例如描述硬件的内部状态。根据传感器的类型，存在一些特定的语义组件，这些组件随 `ros2_controllers` 一起提供广播器。

- [`Imu Sensor Broadcaster`：  `IMU`传感器广播器](https://control.ros.org/humble/doc/ros2_controllers/imu_sensor_broadcaster/doc/userdoc.html#imu-sensor-broadcaster-userdoc)
- [`Force Torque Sensor Broadcaster`：力矩传感器广播器](https://control.ros.org/humble/doc/ros2_controllers/force_torque_sensor_broadcaster/doc/userdoc.html#force-torque-sensor-broadcaster-userdoc)

## GPIOs

`<gpio>`-标签用于描述机器人设备的输入和输出端口，这些端口无法与任何关节或传感器相关联。 `<gpio>` -标签的解析与 `<joint>` -标签类似，具有命令和状态接口。该标签必须至少有一个 `<command>` -或 `<state>` -标签作为子标签。

`gpio`这个关键词因其通用性而被选用。尽管严格来说它用于数字信号，但它描述了任何电气模拟、数字信号或物理值。

`<gpio>`标签可以作为三种硬件组件（系统、传感器或执行器）的子项使用。

由于作为`<gpio>`标签实现的端口通常非常特定于应用，因此在 `ros2_control` 框架中没有通用的发布者。每个应用都需要实现一个自定义的 `gpio` 控制器。例如，请参阅演示仓库中的 `GPIO` [控制器示例](https://control.ros.org/humble/doc/ros2_control_demos/example_10/doc/userdoc.html#ros2-control-demos-example-10-userdoc)。

## 示例

以下示例展示了如何在 `ros2_control` `URDF` 中使用不同的硬件接口类型。它们可以在不同的硬件组件类型（系统、执行器、传感器）中组合使用（参见详细文档），如下所示

### 具有多个 GPIO 接口的机器人

- `RRBot System`
- `Digital: 4 inputs and 2 outputs`(数字：4 个输入和 2 个输出)
- `Analog: 2 inputs and 1 output`(模拟：2 个输入和 1 个输出)
- `Vacuum valve at the flange (on/off)`(法兰上的真空阀（开/关）)

```xml
<ros2_control name="RRBotSystemMutipleGPIOs" type="system">
  <hardware>
    <plugin>ros2_control_demo_hardware/RRBotSystemPositionOnlyHardware</plugin>
    <param name="example_param_hw_start_duration_sec">2.0</param>
    <param name="example_param_hw_stop_duration_sec">3.0</param>
    <param name="example_param_hw_slowdown">2.0</param>
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
  <gpio name="flange_digital_IOs">
    <command_interface name="digital_output1"/>
    <state_interface name="digital_output1"/>    <!-- Needed to know current state of the output -->
    <command_interface name="digital_output2"/>
    <state_interface name="digital_output2"/>
    <state_interface name="digital_input1"/>
    <state_interface name="digital_input2"/>
  </gpio>
  <gpio name="flange_analog_IOs">
    <command_interface name="analog_output1"/>
    <state_interface name="analog_output1">    <!-- Needed to know current state of the output -->
      <param name="initial_value">3.1</param>  <!-- Optional initial value for mock_hardware -->
    </state_interface>
    <state_interface name="analog_input1"/>
    <state_interface name="analog_input2"/>
  </gpio>
  <gpio name="flange_vacuum">
    <command_interface name="vacuum"/>
    <state_interface name="vacuum"/>    <!-- Needed to know current state of the output -->
  </gpio>
</ros2_control>
```

### 具备电气和吸力抓取功能的夹爪

- `Multimodal gripper`: 多模态夹爪
- `1-DoF parallel gripper`: 一自由度并联夹爪
- `suction on/off`: 吸力开/关

```xml
<ros2_control name="MultimodalGripper" type="actuator">
  <hardware>
    <plugin>ros2_control_demo_hardware/MultimodalGripper</plugin>
  </hardware>
  <joint name="parallel_fingers">
    <command_interface name="position">
      <param name="min">0</param>
      <param name="max">100</param>
    </command_interface>
    <state_interface name="position"/>
  </joint>
  <gpio name="suction">
    <command_interface name="suction"/>
    <state_interface name="suction"/>    <!-- Needed to know current state of the output -->
  </gpio>
</ros2_control>
```

### 带温度反馈和可调校准的力-力矩传感器

- `2D FTS`: 2D 力-力矩传感器
- `Temperature feedback in °C`: 温度反馈以°C 为单位
- `Choice between 3 calibration matrices, i.e., calibration ranges`: 在 3 个校准矩阵之间进行选择，即校准范围

```xml
<ros2_control name="RRBotForceTorqueSensor2D" type="sensor">
  <hardware>
    <plugin>ros2_control_demo_hardware/ForceTorqueSensor2DHardware</plugin>
    <param name="example_param_read_for_sec">0.43</param>
  </hardware>
  <sensor name="tcp_fts_sensor">
    <state_interface name="fx"/>
    <state_interface name="tz"/>
    <param name="frame_id">kuka_tcp</param>
    <param name="fx_range">100</param>
    <param name="tz_range">100</param>
  </sensor>
  <sensor name="temp_feedback">
    <state_interface name="temperature"/>
  </sensor>
  <gpio name="calibration">
    <command_interface name="calibration_matrix_nr"/>
    <state_interface name="calibration_matrix_nr"/>
  </gpio>
</ros2_control>
```
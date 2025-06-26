###### datetime:2025/06/25 15:53

###### author:nzb

# Broadcasters(广播器)

广播器用于将硬件组件的传感器数据发布到 `ROS` 主题。在 `ros2_control` 的语境中，广播器仍然是一种控制器，使用与其他控制器相同的控制器接口。

## [力矩传感器广播器(Force Torque Sensor Broadcaster)](https://control.ros.org/humble/doc/ros2_controllers/force_torque_sensor_broadcaster/doc/userdoc.html#force-torque-sensor-broadcaster)

从机器人或传感器的力/力矩状态接口广播消息。发布的消息类型是 `geometry_msgs/msg/WrenchStamped` 。

该控制器是围绕 `ForceTorqueSensor` 语义组件的包装（参见 `controller_interface` 包）。

- [参数](https://control.ros.org/humble/doc/ros2_controllers/force_torque_sensor_broadcaster/doc/userdoc.html#parameters) 

## [IMU 传感器广播器(IMU Sensor Broadcaster)](https://control.ros.org/humble/doc/ros2_controllers/imu_sensor_broadcaster/doc/userdoc.html#imu-sensor-broadcaster)

从 `IMU` 传感器广播消息。发布的消息类型是 `sensor_msgs/msg/Imu`

该控制器是围绕 `IMUSensor` 语义组件的包装（参见 `controller_interface` 包）。

- [参数](https://control.ros.org/humble/doc/ros2_controllers/imu_sensor_broadcaster/doc/userdoc.html#parameters)

## [关节状态广播器(joint_state_broadcaster)](https://control.ros.org/humble/doc/ros2_controllers/joint_state_broadcaster/doc/userdoc.html#joint-state-broadcaster)

广播器读取所有状态接口，并在 `/joint_states` 和 `/dynamic_joint_states` 上报告它们。

- 命令：广播器并非真正的控制器，因此不会接受命令。
- 硬件接口类型：默认情况下，所有可用的关节状态接口都会被使用，除非另有配置。在后一种情况下，结果接口由 `joints` 和 `interfaces` 参数的叉积定义的接口“矩阵”决定。如果某些请求的接口缺失，控制器会打印出警告信息，但其他接口仍能正常工作。如果请求的接口均未定义，控制器在激活时将返回错误。
- [参数](https://control.ros.org/humble/doc/ros2_controllers/joint_state_broadcaster/doc/userdoc.html#parameters)

## [距离传感器广播器(Range Sensor Broadcaster)](https://control.ros.org/humble/doc/ros2_controllers/range_sensor_broadcaster/doc/userdoc.html#range-sensor-broadcaster)

从距离传感器广播消息。发布的消息类型是 `sensor_msgs/msg/Range` 。

该控制器是围绕 `RangeSensor` 语义组件的包装（参见 `controller_interface` 包）。

- [参数](https://control.ros.org/humble/doc/ros2_controllers/range_sensor_broadcaster/doc/userdoc.html#parameters)

## [姿态广播器(Pose Broadcaster)](https://control.ros.org/humble/doc/ros2_controllers/pose_broadcaster/doc/userdoc.html#pose-broadcaster)

用于发布机器人或传感器测量的姿态的广播器。姿态以 `geometry_msgs/msg/PoseStamped` 消息发布，并可选择性地作为 `tf` 变换发布。

控制器是围绕 `PoseSensor` 语义组件的包装（参见 `controller_interface` 包）。

- [参数](https://control.ros.org/humble/doc/ros2_controllers/pose_broadcaster/doc/userdoc.html#parameters)

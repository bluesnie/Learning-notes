###### datetime:2025/02/08 14:37

###### author:nzb

# [创建 MoveIt 插件](https://moveit.picknik.ai/main/doc/examples/creating_moveit_plugins/plugin_tutorial.html#creating-moveit-plugins)

- 运动规划器插件

- 控制器管理器插件示例

`MoveIt` 控制器管理器是您自定义低级控制器的接口，这有点名不副实。 更好的说法是控制器接口。 对于大多数用例来说，如果您的机器人控制器已经提供了用于 `FollowJointTrajectory` 的 ROS 操作，那么附带的 [`MoveItSimpleControllerManager`](https://github.com/moveit/moveit2/blob/main/moveit_plugins/moveit_simple_controller_manager) 就足够了。 如果您使用的是 `ros_control`，那么附带的 [`MoveItRosControlInterface`](https://github.com/moveit/moveit2/blob/main/moveit_plugins/moveit_ros_control_interface) 也是理想之选。 不过，对于某些应用，您可能需要一个更加自定义的控制器管理器。 这里提供了一个用于启动自定义控制器管理器的[示例模板](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/controller_configuration/src/moveit_controller_manager_example.cpp)。

- 约束采样器插件示例(查看官方文档或网上资料)
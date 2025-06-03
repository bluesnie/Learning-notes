###### datetime:2025/05/21 14:30

###### author:nzb

# [调试](https://control.ros.org/humble/doc/ros2_control/doc/debugging.html)

所有控制器和硬件组件都是加载到 `controller_manager` 中的插件。因此，调试器必须连接到 `controller_manager` 。如果你的机器人或机器上运行了多个 `controller_manager` 实例，你需要连接到与你要调试的硬件组件或控制器相关的 `controller_manager` 。

## 如何操作

- 在您的系统上安装 `xterm` 、 `gdb` 和 `gdbserver`：`sudo apt install xterm gdb gdbserver`
- 确保你运行的是“调试”或“带调试信息的发布版本”：这是通过将 `--cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo` 传递给 `colcon build` 完成的。记住，在发布版本中，一些断点可能不会按预期工作，因为相应的行可能已被编译器优化。对于这种情况，建议使用完整的调试版本 ( `--cmake-args -DCMAKE_BUILD_TYPE=Debug` )。
- 修改启动文件以在调试器连接的情况下运行控制器管理器：
  - 版本 A：直接使用 `gdb CLI` 运行：在你的启动文件中的 `controller_manager` 节点条目里添加 `prefix=['xterm -e gdb -ex run --args']` 。由于 `ros2launch` 的工作方式，我们需要在一个单独的终端实例中运行特定的节点。
  - 版本 B：使用 `gdbserver` 运行：在你的启动文件中的 `controller_manager` 节点条目里添加 `prefix=['gdbserver localhost:3000']` 。之后，你可以将 `gdb CLI` 实例或你选择的任何 `IDE` 连接到该 `gdbserver` 实例。确保你从已源码了工作区的终端启动调试器，以便正确解析所有路径。
- 示例启动文件条目:

```python
# Obtain the controller config file for the ros2 control node
controller_config_file = get_package_file("<package name>", "config/controllers.yaml")

controller_manager = Node(
    package="controller_manager",
    executable="ros2_control_node",
    parameters=[controller_config_file],
    output="both",
    emulate_tty=True,
    remappings=[
        ("~/robot_description", "/robot_description")
    ],
    prefix=['xterm -e gdb -ex run --args']  # or prefix=['gdbserver localhost:3000']
)

ld.add_action(controller_manager)
```

## 补充说明

- 调试插件：你只能在插件加载后设置断点。在 `ros2_control` 上下文中，这意味着在控制器/硬件组件加载后：
- 调试构建：通常情况下，仅包含您想要调试的特定软件包的调试信息是实用的。 `colcon build --packages-select [package_name] --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo` 或 `colcon build --packages-select [package_name] --cmake-args -DCMAKE_BUILD_TYPE=Debug`
- 实时：
    - 根据经验，对于实时上下文，使用有意义的日志（需谨慎）或添加额外的调试状态接口（在控制器的情况下为发布者）可能更好。
    - 然而，使用 `gdb` 运行 `controller_manager` 和您的插件仍然对于调试段错误等错误非常有用，因为您可以收集完整的回溯信息。

> 警告：控制器的 `update/on_activate/on_deactivate` 方法以及硬件组件的 `read/write/on_activate/perform_command_mode_switch` 方法都在实时更新循环的上下文中运行。在那里设置断点可能会导致问题，在最坏的情况下甚至可能损坏您的硬件。

## 参考资料

- [ROS 2 和 GDB](https://juraph.com/miscellaneous/ros2_and_gdb/)
- [使用 GDB 调试插件](https://stackoverflow.com/questions/10919832/how-to-use-gdb-to-debug-a-plugin)
- [GDB 命令行教程](https://stackoverflow.com/questions/10919832/how-to-use-gdb-to-debug-a-plugin)

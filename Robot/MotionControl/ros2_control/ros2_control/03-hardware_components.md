###### datetime:2025/05/09 10:46

###### author:nzb

# [硬件组件](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/hardware_components_userdoc.html)

硬件组件表示框架中物理硬件 `ros2_control` 抽象。有三种类型的硬件：执行器、传感器和系统。有关每种类型的详细信息，请查看 [硬件组件](https://control.ros.org/humble/doc/getting_started/getting_started.html#overview-hardware-components) 描述。

## 指南和最佳实践

- 硬件接口类型
- 编写硬件组件
- 不同的更新速率

## 硬件组件的生命周期

方法返回值具有 `rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn` 含义如下：

- `CallbackReturn::SUCCESS`： 方法执行成功。
- `CallbackReturn::FAILURE`： 方法执行失败，生命周期转换不成功。
- `CallbackReturn::ERROR`： 发生了严重错误，应由 `on_error` 方法处理。

在每个方法之后，硬件将转换为以下状态：

- `UNCONFIGURED (on_init, on_cleanup)`:

仅初始化硬件，但未启动通信，并且不会将接口导入 `ResourceManager`。

- `INACTIVE (on_configure, on_deactivate)`:

建立与硬件的通信并配置硬件组件。可以读取状态，并且命令界面（仅限 `System` 和 `Actuator`）可用。

截至目前，硬件组件实现可以继续使用从 `CommandInterfaces` 收到的命令或完全跳过它们。

- `FINALIZED (on_shutdown)`:

硬件接口已准备好卸载/销毁。已分配的内存已清理。

- `ACTIVE (on_activate):  活跃 ( on_activate )`

可以读取状态。

仅限系统和执行器：硬件的电源电路处于激活状态，硬件可以移动，例如，制动器已脱开。命令接口可用，命令应发送到硬件

## 处理在 read() 和 write() 调用期间发生的错误

如果从硬件接口类的 `read()` 或 `write()` 方法返回 `hardware_interface::return_type::ERROR `，则 `on_error(previous_state)` 方法将被调用以处理发生的任何错误

错误处理遵循节点生命周期。如果成功返回 `CallbackReturn::SUCCESS` 并且硬件再次处于 `UNCONFIGURED` 状态，如果发生任何 `ERROR` 或 `FAILURE` ，硬件将结束在 `FINALIZED` 状态并且无法恢复。唯一的选择是重新加载完整的插件，但目前控制器管理器中没有为此提供的服务

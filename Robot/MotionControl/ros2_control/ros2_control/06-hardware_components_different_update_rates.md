###### datetime:2025/05/21 11:39

###### author:nzb

# [硬件组件的不同更新速率](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/different_update_rates_userdoc.html)

## 通过计数循环

[`ros2_control` 主节点](https://github.com/ros-controls/ros2_control/blob/humble/controller_manager/src/ros2_control_node.cpp)的当前实现有一个更新速率，该速率控制了在[hardware_interface(s)](https://github.com/ros-controls/ros2_control/blob/humble/hardware_interface/include/hardware_interface/system_interface.hpp)中 [`read(…)`](https://github.com/ros-controls/ros2_control/blob/0bdcd414c7ab8091f3e1b8d9b73a91c778388e82/hardware_interface/include/hardware_interface/system_interface.hpp#L175)和 [`write(…)`](https://github.com/ros-controls/ros2_control/blob/fe462926416d527d1da163bc3eabd02ee1de9be9/hardware_interface/include/hardware_interface/system_interface.hpp#L178)调用的速率。为了实现您的硬件组件的不同更新速率，您可以使用以下步骤：

- 1、添加主控制环更新率和硬件组件期望的更新率参数

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <xacro:macro name="system_interface" params="name main_loop_update_rate desired_hw_update_rate">

    <ros2_control name="${name}" type="system">
      <hardware>
          <plugin>my_system_interface/MySystemHardware</plugin>
          <param name="main_loop_update_rate">${main_loop_update_rate}</param>
          <param name="desired_hw_update_rate">${desired_hw_update_rate}</param>
      </hardware>
      ...
    </ros2_control>

  </xacro:macro>

</robot>
```

- 2、在你的 `on_init()` 特定实现中获取期望的参数

```c++
namespace my_system_interface
{
hardware_interface::CallbackReturn MySystemHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  //   declaration in *.hpp file --> unsigned int main_loop_update_rate_, desired_hw_update_rate_ = 100 ;
  main_loop_update_rate_ = stoi(info_.hardware_parameters["main_loop_update_rate"]);
  desired_hw_update_rate_ = stoi(info_.hardware_parameters["desired_hw_update_rate"]);

  ...
}
...
} // my_system_interface
```

- 3、在你的 `on_activate` 特定实现中重置内部环计数器

```c++
hardware_interface::CallbackReturn MySystemHardware::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
    //   declaration in *.hpp file --> unsigned int update_loop_counter_ ;
    update_loop_counter_ = 0;
    ...
}
```

- 4、在你的 `read(const rclcpp::Time & time, const rclcpp::Duration & period)` 和/或 `write(const rclcpp::Time & time, const rclcpp::Duration & period)` 特定实现中决定是否应该干扰你的硬件

```c++
hardware_interface::return_type MySystemHardware::read(const rclcpp::Time & time, const rclcpp::Duration & period)
{

  bool hardware_go = main_loop_update_rate_ == 0  ||
                    desired_hw_update_rate_ == 0 ||
                    ((update_loop_counter_ % desired_hw_update_rate_) == 0);

  if (hardware_go){
    // hardware comms and operations
    ...
  }
  ...

  // update counter
  ++update_loop_counter_;
  update_loop_counter_ %= main_loop_update_rate_;
}
```

## 通过测量经过的时间

另一种决定硬件通信是否应在 `read(const rclcpp::Time & time, const rclcpp::Duration & period)` 和/或 `write(const rclcpp::Time & time, const rclcpp::Duration & period)` 实现中执行的方法是测量自上次通过以来的经过时间：

- 1、在您的 `on_activate` 特定实现中重置指示这是 `read` 和 `write` 方法第一次通过的标志

```c++
hardware_interface::CallbackReturn MySystemHardware::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
    //   declaration in *.hpp file --> bool first_read_pass_, first_write_pass_ = true ;
    first_read_pass_ = first_write_pass_ = true;
    ...
}
```

- 2、在您的 `read(const rclcpp::Time & time, const rclcpp::Duration & period)` 和/或 `write(const rclcpp::Time & time, const rclcpp::Duration & period)` 特定实现中决定是否应干扰您的硬件

```c++
hardware_interface::return_type MySystemHardware::read(const rclcpp::Time & time, const rclcpp::Duration & period)
{
    if (first_read_pass_ || (time - last_read_time_ ) > desired_hw_update_period_)
    {
      first_read_pass_ = false;
      //   declaration in *.hpp file --> rclcpp::Time last_read_time_ ;
      last_read_time_ = time;
      // hardware comms and operations
      ...
    }
    ...
}

hardware_interface::return_type MySystemHardware::write(const rclcpp::Time & time, const rclcpp::Duration & period)
{
    if (first_write_pass_ || (time - last_write_time_ ) > desired_hw_update_period_)
    {
      first_write_pass_ = false;
      //   declaration in *.hpp file --> rclcpp::Time last_write_time_ ;
      last_write_time_ = time;
      // hardware comms and operations
      ...
    }
    ...
}
```

> 注意：获取从 `URDF` 中所需的更新周期值并将其分配给变量 `desired_hw_update_period_` 的方法与上一节（步骤 1 和步骤 2）相同，但参数名称不同。

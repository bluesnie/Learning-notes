###### datetime:2025/05/19 16:46

###### author:nzb

# [编写硬件组件](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/writing_new_hardware_component.html)

在 `ros2_control` 硬件系统中，组件是库，由控制器管理器使用 `pluginlib` 接口动态加载。以下是一个逐步指南，用于创建新的硬件接口的源文件、基本测试和编译规则。

- 1、准备软件包

如果硬件接口的软件包不存在，则应先创建它。该软件包应使用 `ament_cmake` 作为构建类型。最简单的方法是搜索最新的手册。一个有助于此过程的命令是 `ros2 pkg create `。使用 `--help` 标志获取更多关于如何使用它的信息。还有一个选项可以创建库源文件和编译规则，以帮助您在后续步骤中。

- 2、准备源文件

创建软件包后，您应该至少有 `CMakeLists.txt` 和 `package.xml` 文件。如果它们不存在，请创建 `include/<PACKAGE_NAME>/` 和 `src` 文件夹。在 `include/<PACKAGE_NAME>/` 文件夹中添加 `<robot_hardware_interface_name>.hpp` 和 `<robot_hardware_interface_name>.cpp` 到 `src` 文件夹中。可选地添加 `visibility_control.h` ，其中包含 `Windows` 导出规则的定义。您可以从现有的控制器软件包中复制此文件，并将名称前缀更改为 `<PACKAGE_NAME>` 。

- 3、将声明添加到头文件 (`.hpp`) 中

  - 注意使用头文件保护。`ROS2` 风格的头文件保护使用 `#ifndef` 和 `#define` 预处理器指令。（有关更多信息，搜索引擎是你的好朋友 :) ）。
  - 如果使用，请包含 `"hardware_interface/$interface_type$_interface.hpp"` 和 `visibility_control.h` 。 `$interface_type$` 可以是 `Actuator` 、 `Sensor` 或 `System` ，具体取决于你使用的硬件类型。有关每种类型的详细信息，请查看硬件[组件描述](https://control.ros.org/humble/doc/getting_started/getting_started.html#overview-hardware-components)。
  - 为你的硬件接口定义一个唯一的命名空间。这通常是写在 `snake_case` 中的包名。
  - 定义硬件接口的类，扩展 `$InterfaceType$Interface` ，例如：`..code:: c++ class HardwareInterfaceName : public hardware_interface::$InterfaceType$Interface`
  - 添加一个无参数的构造函数和以下公共方法实现 `LifecycleNodeInterface` : `on_configure` , `on_cleanup` , `on_shutdown` , `on_activate` , `on_deactivate` , `on_error` ；以及覆盖 `$InterfaceType$Interface` 定义： `on_init` , `export_state_interfaces` , `export_command_interfaces` , `prepare_command_mode_switch` (可选), `perform_command_mode_switch` (可选), `read` , `write` 。有关硬件生命周期的进一步解释，请查看[拉取请求](https://github.com/ros-controls/ros2_control/pull/559/files#diff-2bd171d85b028c1b15b03b27d4e6dcbb87e52f705042bf111840e7a28ab268fc)，有关方法的精确定义，请查看 `"hardware_interface/$interface_type$_interface.hpp"` 头文件或 `Actuator`、`Sensor` 或 `System` 的 `doxygen` [文档](https://control.ros.org/humble/doc/api/namespacehardware__interface.html)。

- 4、在源文件 (`.cpp`) 中添加定义
  
  - 包含硬件接口的头文件，并添加命名空间定义以简化后续开发。
  - 实现 `on_init` 方法。在这里，您应该初始化所有成员变量并处理来自 `info` 参数的参数。通常第一行会调用父类 `on_init` 来处理标准值，如名称。这是通过： `hardware_interface::(Actuator|Sensor|System)Interface::on_init(info)` 完成的。如果所有必需的参数都已设置且有效，并且一切正常，请返回 `CallbackReturn::SUCCESS` ，否则返回`return CallbackReturn::ERROR` 。
  - 编写 `on_configure` 方法，通常用于设置与硬件的通信，并设置所有内容，以便硬件可以被激活。
  - 实现 `on_cleanup` 方法，该方法的操作与 `on_configure` 相反。
  - 实现 `export_state_interfaces` 和 `export_command_interfaces` 方法，其中定义了硬件提供的接口。对于 `Sensor` 类型的硬件接口，没有 `export_command_interfaces` 方法。作为提醒，完整的接口名称结构为 `<joint_name>/<interface_type>` 。
  - （可选）对于执行器和系统类型的硬件接口，如果您的硬件接受多种控制模式，请实现 `prepare_command_mode_switch` 和 `perform_command_mode_switch` 。
  - 实现 `on_activate` 方法，启用硬件“电源”。
  - 实现 `on_deactivate` 方法，其作用与 `on_activate` 相反。
  - 实现关闭硬件的优雅方法。
  - 实现 `on_error` 方法，处理所有状态的不同错误。
  - 实现从硬件获取状态并将它们存储在 `export_state_interfaces` 中定义的内部变量中的 `read` 方法。
  - 实现 `write` 方法，根据在 `export_command_interfaces` 中定义的内部变量存储的值来控制硬件。
  - 重要提示：在文件中的命名空间关闭后，添加 `PLUGINLIB_EXPORT_CLASS` 宏。为此，您需要包含 `"pluginlib/class_list_macros.hpp"` 头文件。作为第一个参数，您应该提供确切的硬件接口类，例如 `<my_hardware_interface_package>::<RobotHardwareInterfaceName>` ，作为第二个参数提供基类，即 `hardware_interface::(Actuator|Sensor|System)Interface` 。
- 5、为 `pluginlib` 编写导出定义

  - 在包中创建 `<my_hardware_interface_package>.xml` 文件，并添加库和硬件接口类的定义，该类必须对 `pluginlib` 可见。最简单的方法是查看 [`hardware_interface`](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/mock_components_userdoc.html#mock-components-userdoc) [`mock_components`](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/mock_components_userdoc.html#mock-components-userdoc) 部分的定义。
  - 通常，插件名称由包（命名空间）和类名定义，例如 `<my_hardware_interface_package>/<RobotHardwareInterfaceName>` 。此名称定义了资源管理器搜索时的硬件接口类型。其他两个参数必须与 `<robot_hardware_interface_name>.cpp` 文件底部的宏中的定义相对应。

- 6、编写一个简单的测试来检查控制器是否可以被找到和加载

  - 在您的软件包中创建文件夹 `test` ，如果它尚不存在，并添加一个名为 `test_load_<robot_hardware_interface_name>.cpp` 的文件。
  - 您可以复制在 [`test_generic_system.cpp`](https://github.com/ros-controls/ros2_control/blob/humble/hardware_interface/test/mock_components/test_generic_system.cpp#L402-L407) 软件包中定义的 `load_generic_system_2dof` 内容。
  - 更改复制的测试名称，并在最后一行，指定硬件接口类型的地方，放入 `<my_hardware_interface_package>.xml` 文件中定义的名称，例如 `<my_hardware_interface_package>/<RobotHardwareInterfaceName>` 。

- 7、在 `CMakeLists.txt` 文件中添加编译指令。

  - 在 `find_package(ament_cmake REQUIRED)` 这一行下方添加进一步的依赖项。这些至少包括： `hardware_interface` 、 `pluginlib` 、 `rclcpp` 和 `rclcpp_lifecycle` 。
  - 为一个提供 `<robot_hardware_interface_name>.cpp` 文件作为源代码的共享库添加编译指令。
  - 添加库的定向包含目录。这通常只是 `include` 。
  - 添加库所需的 `ament` 依赖。至少应添加 1 中列出的依赖项。。
  - 使用以下命令导出用于 `pluginlib` 的描述文件：`.. code:: cmake`    
    `pluginlib_export_plugin_description_file(hardware_interface <my_hardware_interface_package>.xml)`
  - 添加目标和包含目录的安装指令。
  - 在测试部分添加以下依赖： `ament_cmake_gmock` ， `hardware_interface` 。
  - 使用 `ament_add_gmock` 指令为测试添加编译定义。详细信息，请参考如何在 `ros2_control` 软件包中为模拟硬件进行操作。
  - （可选）在 `ament_export_libraries` 之前将您的硬件接口库添加到 `ament_package()` 。

- 8、在 `package.xml` 文件中添加依赖项

  - 在 `<depend>` 标签中至少添加以下软件包： `hardware_interface` 、 `pluginlib` 、 `rclcpp` 和 `rclcpp_lifecycle` 。
  - 在 `<test_depend>` 标签中至少添加以下软件包： `ament_add_gmock` 和 `hardware_interface` 

- 9、编译和测试硬件组件

  - 现在一切就绪，可以使用 `colcon build <my_hardware_interface_package>` 命令编译硬件组件。请记住在执行此命令之前进入工作区的根目录。
  - 如果编译成功，从安装文件夹中 `source setup.bash` 文件，并执行 `colcon test <my_hardware_interface_package>` 来检查新的控制器是否可以通过 `pluginlib` 库找到并由控制器管理器加载。
###### datetime:2025/06/23 11:50

###### author:nzb

# [编写新的控制器](https://control.ros.org/humble/doc/ros2_controllers/doc/writing_new_controller.html)

- 准备软件包：`ros2 pkg create`

- 准备源文件

    创建包后，你应该至少有 `CMakeLists.txt` 和 `package.xml` 这两个文件。如果 `include/<PACKAGE_NAME>`/ 和 `src` 文件夹不存在，请创建它们。在 `include/<PACKAGE_NAME>/` 文件夹中，将 `<controller_name>.hpp` 和 `<controller_name>.cpp` 文件添加到 `src` 文件夹中。可选地，添加 `visibility_control.h` 文件，其中包含 `Windows` 导出规则的定义。你可以从现有的控制器包中复制这个文件，并将文件名前缀更改为 `<PACKAGE_NAME>` 。

- 将声明添加到头文件 (`.hpp`)
    - 注意使用头文件保护。`ROS2` 风格使用 `#ifndef` 和 `#define` 预处理器指令。（有关更多信息，搜索引擎是你的好朋友 :)）。
    - 如果你在使用，请包含 `"controller_interface/controller_interface.hpp"` 和 `visibility_control.h` 。
    - 为你的控制器定义一个唯一的命名空间。这通常是一个用 `snake_case` 写的包名。
    - 定义控制器的类，扩展 `ControllerInterface` ，例如，`class ControllerName : public controller_interface::ControllerInterface`
    - 添加一个无参数的构造函数和以下覆盖 `ControllerInterface` 定义的公共方法： `on_init` 、 `command_interface_configuration` 、 `state_interface_configuration` 、 `on_configure` 、 `on_activate` 、 `on_deactivate` 、 `update` 。确切定义请查看 `controller_interface/controller_interface.hpp` 头文件或 [`ros2_controllers`](https://github.com/ros-controls/ros2_controllers) 中的任何一个控制器。
    - （可选）通常，控制器接受关节名称和接口名称的列表作为参数。如果是这样，你可以添加两个受保护的字符串向量来存储这些值。

- 向源文件（`.cpp`）中添加定义
    - 包含你的控制器的头文件，并添加一个命名空间定义以简化后续开发。
    - （可选）如果需要，实现一个构造函数。在那里，你可以初始化成员变量。这也可以在 `on_init` 方法中完成。
    - 实现 `on_init` 方法。第一行通常调用父类 `on_init` 方法。这里是最适合初始化变量、预留内存，以及最重要的是，声明控制器使用的节点参数的地方。如果一切正常，返回 `controller_interface::return_type::OK` 或 `controller_interface::return_type::ERROR` ，否则。
    - 编写 `on_configure` 方法。参数通常在这里读取，并且所有准备工作都完成，以便控制器可以启动。
    - 实现 `command_interface_configuration` 和 `state_interface_configuration` ，其中所需接口定义。在 `controller_interface/controller_interface.hpp` 中定义了三种接口配置选项 `ALL` 、 `INDIVIDUAL` 和 `NONE` 。 `ALL` 和 `NONE` 选项将要求访问所有可用接口或没有任何接口。 `INDIVIDUAL` 配置需要一个详细所需接口名称的列表。这些通常作为参数提供。完整接口名称具有结构 `<joint_name>/<interface_type>` 。
    - 实现带检查和潜在排序接口的 `on_activate` 方法，并分配成员的初始值。此方法是实时循环的一部分，因此避免任何内存预留，并且一般来说，尽可能保持其尽可能短。
    - 实现 `on_deactivate` 方法，该方法与 `on_activate` 相反。在许多情况下，此方法为空。此方法应尽可能保证实时安全。
    - 将 `update` 方法实现为主入口点。该方法应考虑实时约束。当调用此方法时，状态接口拥有来自硬件的最新值，硬件的新命令应写入命令接口。
    - 重要提示：在文件中命名空间关闭后，添加 `PLUGINLIB_EXPORT_CLASS` 宏。为此，您需要包含 `"pluginlib/class_list_macros.hpp"` 头文件。作为第一个参数，应提供确切的控制器类，例如 `<controller_name_namespace>::<ControllerName>` ，作为第二个参数提供基类，即 `controller_interface::ControllerInterface` 。

- 为 `pluginlib` 编写导出定义
    - 在包中创建 `<controller_name>.xml` 文件，并添加库和控制器类的定义，该定义需要对 `pluginlib` 可见。最简单的方法是查看 [`ros2_controllers`](https://github.com/ros-controls/ros2_controllers) 包中的其他控制器。
    - 通常，插件名称由包（命名空间）和类名定义，例如 `<controller_name_package>/<ControllerName>` 。该名称在控制器管理器搜索控制器时定义了控制器的类型。另外两个参数必须与 `<controller_name>.cpp` 文件底部的宏定义相匹配。

- 编写简单的测试来检查控制器是否可以被找到并加载
    - 如果您的包中不存在 `test` 文件夹，请创建该文件夹，并添加一个名为 `test_load_<controller_name>.cpp` 的文件。
    - 你可以安全地复制文件内容，用于在 [`ros2_controllers`](https://github.com/ros-controls/ros2_controllers) 包中定义的任何控制器。
    - 将复制的测试文件重命名，并在最后一行指定控制器类型的地方，放入 `<controller_name>.xml` 文件中定义的名称，例如 `<controller_name_package>/<ControllerName>` 。

- 在`CMakeLists.txt`文件中添加编译指令
    - 在 `find_package(ament_cmake REQUIRED)` 行下方添加其他依赖项。这些至少包括： `controller_interface` 、 `hardware_interface` 、 `pluginlib` 、 `rclcpp` 和 `rclcpp_lifecycle` 。
    - 为共享库添加编译指令，将 `<controller_name>.cpp` 文件作为源文件。
    - 为目标库添加包含目录。这通常只有 `include` 。
    - 添加库所需的 `ament` 依赖项。你应该至少添加 1 下表中列出的那些。
    - 使用以下命令为 `pluginlib` 描述文件导出：`pluginlib_export_plugin_description_file(controller_interface <controller_name>.xml)`
    - 为目标和包含目录添加安装指令。
    - 在测试部分添加以下依赖： `ament_cmake_gmock` 、 `controller_manager` 、 `hardware_interface` 、 `ros2_control_test_assets` 。
    - 使用 `ament_add_gmock` 指令为测试添加编译定义。具体方法，请参考 `ros2_controllers` 包中控制器的配置方式。
    - （可选）在 `ament_export_libraries` 之前将您的控制器库添加到 `ament_package()` 。

- 在 `package.xml` 文件中添加依赖。
    - 在 `<depend>` 标签中至少添加以下包： `controller_interface` 、 `hardware_interface` 、 `pluginlib` 、 `rclcpp` 和 `rclcpp_lifecycle` 。
    - 在 `<test_depend>` 标签中至少添加以下包： `ament_add_gmock` 、 `controller_manager` 、 `hardware_interface` 和 `ros2_control_test_assets` 。

- 编译和测试控制器
    - 现在一切就绪，可以使用 `colcon build <controller_name_package>` 命令来编译控制器。执行此命令前，请确保已进入工作空间的根目录。
    - 如果编译成功，从安装文件夹中 `source setup.bash` 文件，并执行 `colcon test <controller_name_package>` 来检查新控制器是否可通过 `pluginlib` 库找到并由控制器管理器加载。

###### datetime:2025/01/21 16:22

###### author:nzb

# [入门](https://moveit.picknik.ai/main/doc/tutorials/getting_started/getting_started.html)

## 1.1 安装

- [安装教程](https://moveit.picknik.ai/main/doc/tutorials/getting_started/getting_started.html#install-ros-2-and-colcon)

> 注意：使用此命令`colcon build --mixin release`构建的某些软件包需要高达 `16GB` 的 RAM 才能构建。默认情况下，`colcon` 会尝试同时构建尽可能多的软件包。如果您的计算机内存不足，或者构建通常无法在您的计算机上完成，您可以尝试将 `--executor equation` 附加到上面的 `colcon` 命令以一次仅构建一个软件包，或 `--parallel-workers <X>` 以限制同时构建的数量。对于更受限制的机器，您可以尝试运行 `MAKEFLAGS="-j4 -l1" colcon build --executor equation`。
>
> 本机使用``MAKEFLAGS="-j4 -l1" colcon build --executor equation`

## 1.2 [RViz 中的 MoveIt 快速入门](https://moveit.picknik.ai/main/doc/tutorials/quickstart_in_rviz/quickstart_in_rviz_tutorial.html#moveit-quickstart-in-rviz)

### 1.2.1 启动

```shell
ros2 launch moveit2_tutorials demo.launch.py
```

### 1.2.2 玩转可视化机器人
有四种不同的重叠可视化效果： 
- `/monitored_planning_scene`规划环境中的机器人配置（默认为激活）。 
- 机器人的规划路径（默认为激活）。 
- 绿色：运动规划的开始状态（默认为禁用）。
- 橙色：运动规划的目标状态（默认为激活）。

每种可视化的显示状态都可以通过复选框进行切换：

- 使用 `Scene Robot`树形菜单中的 `Show Robot Visual`复选框显示规划场景机器人。 
- 使用 `Planned Path`树形菜单中的 `Show Robot Visual`复选框显示规划路径。 
- 使用 `Planning Request`树形菜单中的 `Query Start State`复选框显示起始状态。 
- 使用 `Planning Request`树形菜单中的 `Query Goal State`复选框显示目标状态。

### 1.2.3 与 Kinova Gen 3 互动
- 进入碰撞
- 移到可达的工作空间
- 移动关节或无效空间

### 1.2.4 与 Kinova Gen 3 一起使用运动规划
- 审视轨迹航点
- 平面直角坐标运动（直线规划）
- 执行轨迹，调整速度
  
## 1.3 [第一个C++程序](https://moveit.picknik.ai/main/doc/tutorials/your_first_project/your_first_project.html#your-first-c-moveit-project)

### 1.3.1 创建项目
```shell
ros2 pkg create \
 --build-type ament_cmake \
 --dependencies moveit_ros_planning_interface rclcpp \
 --node-name hello_moveit hello_moveit
```

### 1.3.2 编写代码

```c++
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

int main(int argc, char *argv[])
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
      "hello_moveit", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // Create a ROS logger
  auto const logger = rclcpp::get_logger("hello_moveit");

  // Create the MoveIt MoveGroup Interface
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "panda_arm");

  // Set a target Pose, 匿名函数
  auto const target_pose = []
  {
    geometry_msgs::msg::Pose msg;
    msg.orientation.w = 1.0;
    msg.position.x = 0.28;
    msg.position.y = -0.2;
    msg.position.z = 0.5;
    return msg;
  }();
  move_group_interface.setPoseTarget(target_pose);  // 可以通过 MoveGroupInterface::setStartState* 设置起始状态

  // Create a plan to that target pose
  auto const [success, plan] = [&move_group_interface]
  {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  // Execute the plan
  if (success)
  {
    move_group_interface.execute(plan);
  }
  else
  {
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  // Shutdown ROS
  rclcpp::shutdown();
  return 0;
}
```

### 1.3.3 编译运行

- `colcon build --mixin debug`
- `ros2 launch moveit2_tutorials demo.launch.py`
- 另一个终端 `ros2 run hello_moveit hello_moveit`

- 问题
  - 请注意，如果在未启动演示启动文件的情况下运行 hello_moveit 节点，它会等待 10 秒，然后打印此错误并退出。
    
    ```shell
    [ERROR] [1644181704.350825487] [hello_moveit]: Could not find parameter robot_description and did not receive robot_description via std_msgs::msg::String subscription within 10.000000 seconds.
    ```

  - 这是因为 `demo.launch.py` 启动的是提供机器人描述的 `MoveGroup` 节点。 在构建` MoveGroupInterface` 时，它会查找发布机器人描述主题的节点。 如果在 `10` 秒内没有找到，它就会打印此错误并终止程序。

### 1.3.4 更多阅读
- 我们使用 `lambdas`匿名函数可以将对象初始化为常量。 这就是所谓的 IIFE 技术。 [请阅读 C++ Stories 中有关这种模式的更多信息。](https://www.cppstories.com/2016/11/iife-for-complex-initialization/)
- 我们还将所有可以声明的内容都声明为 const。 [点击此处了解更多 const 的用处。](https://www.cppstories.com/2016/12/please-declare-your-variables-as-const/)
  - 性能高？
  - 变量声明为本地使用
  - 明确意图(目前该变量不会被修改)
  - 代码简洁
  - bug少，不会因为变量被意外修改而引起bug
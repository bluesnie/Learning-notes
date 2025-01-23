###### datetime:2025/01/21 17:22

###### author:nzb

# [在 RViz 中可视化调试](https://moveit.picknik.ai/main/doc/tutorials/visualizing_in_rviz/visualizing_in_rviz.html)

## 1.1 添加依赖
- `package.xml` 文件中添加内容`<depend>moveit_visual_tools</depend>`
- `CMakeLists.txt` 文件中添加内容

```cmake
find_package(moveit_visual_tools REQUIRED)

ament_target_dependencies(
  hello_moveit
  "moveit_ros_planning_interface"
  "moveit_visual_tools"
  "rclcpp"
)
```

## 1.2 编写代码

```cpp
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

int main(int argc, char *argv[])
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
      "hello_moveit", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // Create a ROS logger
  auto const logger = rclcpp::get_logger("hello_moveit");

  // We spin up a SingleThreadedExecutor for the current state monitor to get
  // information about the robot's state.
  // 为 MoveItVisualTools 启动一个 SingleThreadedExecutor，以便与 ROS 交互
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]()
                             { executor.spin(); });

  // Create the MoveIt MoveGroup Interface
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "panda_arm");

  // Construct and initialize MoveItVisualTools
  auto moveit_visual_tools =
      moveit_visual_tools::MoveItVisualTools{node, "panda_link0", rviz_visual_tools::RVIZ_MARKER_TOPIC,
                                             move_group_interface.getRobotModel()};
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.loadRemoteControl();

  // Create a closure for updating the text in rviz
  // 在机器人底座上方一米处添加文字
  auto const draw_title = [&moveit_visual_tools](auto text)
  {
    auto const text_pose = []
    {
      auto msg = Eigen::Isometry3d::Identity();
      msg.translation().z() = 1.0; // Place text 1m above the base link
      return msg;
    }();
    moveit_visual_tools.publishText(text_pose, text, rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
  };
  // 在用户按下 RViz 中的下一个按钮之前，该函数会阻塞程序
  auto const prompt = [&moveit_visual_tools](auto text)
  { moveit_visual_tools.prompt(text); };
  // 绘制我们规划的轨迹路径
  auto const draw_trajectory_tool_path =
      [&moveit_visual_tools, jmg = move_group_interface.getRobotModel()->getJointModelGroup("panda_arm")](
          auto const trajectory)
  { moveit_visual_tools.publishTrajectoryLine(trajectory, jmg); };

  // Set a target Pose
  auto const target_pose = []
  {
    geometry_msgs::msg::Pose msg;
    msg.orientation.w = 1.0;
    msg.position.x = 0.28;
    msg.position.y = -0.2;
    msg.position.z = 0.5;
    return msg;
  }();
  move_group_interface.setPoseTarget(target_pose);

  // Create a plan to that target pose
  prompt("Press 'next' in the RvizVisualToolsGui window to plan");
  draw_title("Planning");
  moveit_visual_tools.trigger(); // 发送到 RViz 的消息会在调用 trigger 时批量发送，以减少标记主题的带宽。
  auto const [success, plan] = [&move_group_interface]
  {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  // Execute the plan
  if (success)
  {
    draw_trajectory_tool_path(plan.trajectory_);
    moveit_visual_tools.trigger();
    prompt("Press 'next' in the RvizVisualToolsGui window to execute");
    draw_title("Executing");
    moveit_visual_tools.trigger();
    move_group_interface.execute(plan);
  }
  else
  {
    draw_title("Planning Failed!");
    moveit_visual_tools.trigger();
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  // Shutdown ROS
  rclcpp::shutdown();
  spinner.join();
  return 0;
}
``` 
 
 ### 1.3 编译运行

- `colcon build --mixin debug`
- [修改RViz可视化](https://moveit.picknik.ai/main/doc/tutorials/visualizing_in_rviz/visualizing_in_rviz.html#enable-visualizations-in-rviz)
- `ros2 launch moveit2_tutorials demo.launch.py`
- 另一个终端 `ros2 run hello_moveit hello_moveit`

### 1.4 更多阅读
- MoveItVisualTools 还有更多用于机器人运动可视化的实用功能。 [点击此处了解更多信息。](https://github.com/moveit/moveit_visual_tools/tree/ros2)
- [MoveItCpp 教程中还有更多使用 `MoveItVisualTools` 的示例。](https://moveit.picknik.ai/main/doc/examples/moveit_cpp/moveitcpp_tutorial.html)
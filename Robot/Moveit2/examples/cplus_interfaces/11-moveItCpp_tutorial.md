###### datetime:2025/02/14 17:28

###### author:nzb

# [MoveIt CPP Tutorial](https://moveit.picknik.ai/main/doc/examples/moveit_cpp/moveitcpp_tutorial.html)

`MoveItCpp` 是一种新的高级接口，是一种统一的 C++ 应用程序接口，无需使用 ROS 操作、服务和消息即可访问 `MoveIt` 核心功能，是现有 `MoveGroup` 应用程序接口的替代品（而非完全替代）。 该接口是 `PickNik Robotics` 为众多商业应用而开发的。

## 运行

```bash
ros2 launch moveit2_tutorials moveit_cpp_tutorial.launch.py
```

## 代码解读

```cpp
#include <rclcpp/rclcpp.hpp>
#include <memory>
// MoveitCpp
#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/moveit_cpp/planning_component.h>

#include <geometry_msgs/msg/point_stamped.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

namespace rvt = rviz_visual_tools;

// 所有使用ROS日志记录的源文件应在文件顶部定义一个名为LOGGER的文件特定静态常量rclcpp::Logger，位于具有最窄范围的命名空间中（如果有）
static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_cpp_tutorial");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  RCLCPP_INFO(LOGGER, "Initialize node");

  // 这可以加载未声明的参数，最佳实践是在相应的类中声明参数并提供有关预期使用的描述
  node_options.automatically_declare_parameters_from_overrides(true);
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("run_moveit_cpp", "", node_options);

  // 我们为当前状态监视器启动一个SingleThreadedExecutor，以获取有关机器人状态的信息。
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // BEGIN_TUTORIAL
  //
  // Setup
  // ^^^^^
  //
  static const std::string PLANNING_GROUP = "panda_arm";
  static const std::string LOGNAME = "moveit_cpp_tutorial";

  /* Otherwise robot with zeros joint_states */
  // 否则，机器人将具有零关节状态
  rclcpp::sleep_for(std::chrono::seconds(1));

  RCLCPP_INFO(LOGGER, "Starting MoveIt Tutorials...");

  auto moveit_cpp_ptr = std::make_shared<moveit_cpp::MoveItCpp>(node);
  moveit_cpp_ptr->getPlanningSceneMonitor()->providePlanningSceneService();

  auto planning_components = std::make_shared<moveit_cpp::PlanningComponent>(PLANNING_GROUP, moveit_cpp_ptr);
  auto robot_model_ptr = moveit_cpp_ptr->getRobotModel();
  auto robot_start_state = planning_components->getStartState();
  auto joint_model_group_ptr = robot_model_ptr->getJointModelGroup(PLANNING_GROUP);

  // Visualization
  // ^^^^^^^^^^^^^
  //
  // The package MoveItVisualTools provides many capabilities for visualizing objects, robots, and trajectories in RViz
  // as well as debugging tools such as step-by-step introspection of a script
  // 翻译：MoveItVisualTools软件包提供了在RViz中可视化对象、机器人和轨迹以及调试工具（如脚本单步调试）等多种功能
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "panda_link0", "moveit_cpp_tutorial",
                                                      moveit_cpp_ptr->getPlanningSceneMonitor());
  visual_tools.deleteAllMarkers();
  visual_tools.loadRemoteControl();

  Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
  text_pose.translation().z() = 1.75;
  visual_tools.publishText(text_pose, "MoveItCpp_Demo", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  // Start the demo
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  // Planning with MoveItCpp
  // ^^^^^^^^^^^^^^^^^^^^^^^
  // There are multiple ways to set the start and the goal states of the plan they are illustrated in the following plan
  // examples
  // 有多种方法可以设置计划的起始和目标状态，它们在以下计划示例中进行了说明
  //
  // Plan #1：设置起始状态为当前状态
  // ^^^^^^^
  //
  // We can set the start state of the plan to the current state of the robot
  // 我们可以将规划的起始状态设置为机器人的当前状态
  planning_components->setStartStateToCurrentState();

  // The first way to set the goal of the plan is by using geometry_msgs::PoseStamped ROS message type as follow
  // 设置计划目标的第一种方法是使用geometry_msgs::PoseStamped ROS消息类型，如下所示
  geometry_msgs::msg::PoseStamped target_pose1;
  target_pose1.header.frame_id = "panda_link0";
  target_pose1.pose.orientation.w = 1.0;
  target_pose1.pose.position.x = 0.28;
  target_pose1.pose.position.y = -0.2;
  target_pose1.pose.position.z = 0.5;
  planning_components->setGoal(target_pose1, "panda_link8");

  // 现在，我们调用PlanningComponents来计算计划并可视化它。
  // Note that we are just planning
  auto plan_solution1 = planning_components->plan();

  // 检查PlanningComponents是否成功找到计划
  if (plan_solution1)
  {
    // Visualize the start pose in Rviz
    visual_tools.publishAxisLabeled(robot_start_state->getGlobalLinkTransform("panda_link8"), "start_pose");
    // Visualize the goal pose in Rviz
    visual_tools.publishAxisLabeled(target_pose1.pose, "target_pose");
    visual_tools.publishText(text_pose, "setStartStateToCurrentState", rvt::WHITE, rvt::XLARGE);
    // Visualize the trajectory in Rviz
    visual_tools.publishTrajectoryLine(plan_solution1.trajectory, joint_model_group_ptr);
    visual_tools.trigger();

    /* Uncomment if you want to execute the plan */
    // 如果您想执行规划，请取消注释
    /* planning_components->execute(); // Execute the plan */
  }

  // Plan #1 visualization:
  // Start the next plan
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  // Plan #2：使用moveit::core::RobotState设置规划的当前状态
  // ^^^^^^^
  //
  // Here we will set the current state of the plan using moveit::core::RobotState
  // 我们将使用moveit::core::RobotState设置规划的当前状态
  auto start_state = *(moveit_cpp_ptr->getCurrentState());
  geometry_msgs::msg::Pose start_pose;
  start_pose.orientation.w = 1.0;
  start_pose.position.x = 0.55;
  start_pose.position.y = 0.0;
  start_pose.position.z = 0.6;

  start_state.setFromIK(joint_model_group_ptr, start_pose);

  planning_components->setStartState(start_state);

  // 我们将重用我们之前的目标并规划到它。
  auto plan_solution2 = planning_components->plan();
  if (plan_solution2)
  {
    moveit::core::RobotState robot_state(robot_model_ptr);
    moveit::core::robotStateMsgToRobotState(plan_solution2.start_state, robot_state);

    visual_tools.publishAxisLabeled(robot_state.getGlobalLinkTransform("panda_link8"), "start_pose");
    visual_tools.publishAxisLabeled(target_pose1.pose, "target_pose");
    visual_tools.publishText(text_pose, "moveit::core::RobotState_Start_State", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(plan_solution2.trajectory, joint_model_group_ptr);
    visual_tools.trigger();

    /* Uncomment if you want to execute the plan */
    /* planning_components->execute(); // Execute the plan */
  }

  // Plan #2 visualization:
  // Start the next plan
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  // Plan #3 ：使用moveit::core::RobotState设置规划的终点
  // ^^^^^^^
  //
  // 我们还可以使用moveit::core::RobotState设置规划的终点
  auto target_state = *robot_start_state;
  geometry_msgs::msg::Pose target_pose2;
  target_pose2.orientation.w = 1.0;
  target_pose2.position.x = 0.55;
  target_pose2.position.y = -0.05;
  target_pose2.position.z = 0.8;

  target_state.setFromIK(joint_model_group_ptr, target_pose2);

  planning_components->setGoal(target_state);

  // We will reuse the old start that we had and plan from it.
  auto plan_solution3 = planning_components->plan();
  if (plan_solution3)
  {
    moveit::core::RobotState robot_state(robot_model_ptr);
    moveit::core::robotStateMsgToRobotState(plan_solution3.start_state, robot_state);

    visual_tools.publishAxisLabeled(robot_state.getGlobalLinkTransform("panda_link8"), "start_pose");
    visual_tools.publishAxisLabeled(target_pose2, "target_pose");
    visual_tools.publishText(text_pose, "moveit::core::RobotState_Goal_Pose", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(plan_solution3.trajectory, joint_model_group_ptr);
    visual_tools.trigger();

    /* Uncomment if you want to execute the plan */
    /* planning_components->execute(); // Execute the plan */
  }

  // Plan #3 visualization:
  // Start the next plan
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  // Plan #4：使用组状态设置
  // ^^^^^^^
  //
  // We can set the start state of the plan to the current state of the robot
  // 我们可以将规划的起点设置为机器人的当前状态
  // We can set the goal of the plan using the name of a group states
  // 我们可以使用组状态的名称设置规划的终点
  // for panda robot we have one named robot state for "panda_arm" planning group called "ready"
  // 对于熊猫机器人，我们有一个名为“panda_arm”规划组的名为“ready”的组状态
  // see `panda_arm.xacro
  // <https://github.com/ros-planning/moveit_resources/blob/ros2/panda_moveit_config/config/panda_arm.xacro#L13>`_

  /* // Set the start state of the plan from a named robot state */
  /* planning_components->setStartState("ready"); // Not implemented yet */
  // Set the goal state of the plan from a named robot state
  planning_components->setGoal("ready");

  // Again we will reuse the old start that we had and plan from it.
  auto plan_solution4 = planning_components->plan();
  if (plan_solution4)
  {
    moveit::core::RobotState robot_state(robot_model_ptr);
    moveit::core::robotStateMsgToRobotState(plan_solution4.start_state, robot_state);

    visual_tools.publishAxisLabeled(robot_state.getGlobalLinkTransform("panda_link8"), "start_pose");
    visual_tools.publishAxisLabeled(robot_start_state->getGlobalLinkTransform("panda_link8"), "target_pose");
    visual_tools.publishText(text_pose, "Goal_Pose_From_Named_State", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(plan_solution4.trajectory, joint_model_group_ptr);
    visual_tools.trigger();

    /* Uncomment if you want to execute the plan */
    /* planning_components->execute(); // Execute the plan */
  }

  // Plan #4 visualization:
  // Start the next plan
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");
  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  // Plan #5
  // ^^^^^^^
  //
  // 我们还可以生成碰撞场景中物体的运动规划。
  //
  // First we create the collision object
  moveit_msgs::msg::CollisionObject collision_object;
  collision_object.header.frame_id = "panda_link0";
  collision_object.id = "box";

  shape_msgs::msg::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions = { 0.1, 0.4, 0.1 };

  geometry_msgs::msg::Pose box_pose;
  box_pose.position.x = 0.4;
  box_pose.position.y = 0.0;
  box_pose.position.z = 1.0;

  collision_object.primitives.push_back(box);
  collision_object.primitive_poses.push_back(box_pose);
  collision_object.operation = collision_object.ADD;

  // Add object to planning scene
  {  // Lock PlanningScene
    planning_scene_monitor::LockedPlanningSceneRW scene(moveit_cpp_ptr->getPlanningSceneMonitor());
    scene->processCollisionObjectMsg(collision_object);
  }  // Unlock PlanningScene
  planning_components->setStartStateToCurrentState();
  planning_components->setGoal("extended");

  auto plan_solution5 = planning_components->plan();
  if (plan_solution5)
  {
    visual_tools.publishText(text_pose, "Planning_Around_Collision_Object", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(plan_solution5.trajectory, joint_model_group_ptr);
    visual_tools.trigger();

    /* Uncomment if you want to execute the plan */
    /* planning_components->execute(); // Execute the plan */
  }

  // Plan #5 visualization:
  // END_TUTORIAL
  visual_tools.prompt("Press 'next' to end the demo");
  visual_tools.deleteAllMarkers();
  visual_tools.trigger();

  RCLCPP_INFO(LOGGER, "Shutting down.");
  rclcpp::shutdown();
  return 0;
}
```
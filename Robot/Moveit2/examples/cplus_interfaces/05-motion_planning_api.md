###### datetime:2025/01/27 11:26

###### author:nzb

# [运动规划接口](https://moveit.picknik.ai/main/doc/examples/motion_planning_api/motion_planning_api_tutorial.html#motion-planning-api)

## 功能目录
- 运动规划可视化
- 设置运动规划目标位姿
- 可视化结果
- 关节空间目标
- 添加路径约束

## 代码解读

```c++
#include <pluginlib/class_loader.hpp>

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/move_group_interface/move_group_interface.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("motion_planning_api_tutorial");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  std::shared_ptr<rclcpp::Node> motion_planning_api_tutorial_node = rclcpp::Node::make_shared("motion_planning_api_tutorial", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(motion_planning_api_tutorial_node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // BEGIN_TUTORIAL
  // Start
  // ^^^^^
  // 设置好开始使用规划器就很简单了。规划器在MoveIt中作为插件设置，你可以使用ROS插件库接口来加载你想要使用的任何规划器。
  // 在我们加载规划器之前，我们需要两个对象，一个RobotModel和一个PlanningScene。我们将首先实例化一个RobotModelLoader对象，它将在ROS参数服务器上查找机器人描述，并为我们构建一个RobotModel来使用。
  const std::string PLANNING_GROUP = "panda_arm";
  robot_model_loader::RobotModelLoader robot_model_loader(motion_planning_api_tutorial_node, "robot_description");
  const moveit::core::RobotModelPtr& robot_model = robot_model_loader.getModel();
  // 创建一个RobotState和一个JointModelGroup来跟踪当前机器人的姿态和规划组
  moveit::core::RobotStatePtr robot_state(new moveit::core::RobotState(robot_model));
  const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);

  // 我们可以使用RobotModel来构建一个PlanningScene来维护世界状态（包括机器人）。
  planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));

  // 配置一个有效的机器人状态
  planning_scene->getCurrentStateNonConst().setToDefaultValues(joint_model_group, "ready");

  // 我们将现在构建一个加载器，按名称加载规划器。注意我们在这里使用ROS插件库库。
  std::unique_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> planner_plugin_loader;
  planning_interface::PlannerManagerPtr planner_instance;
  std::vector<std::string> planner_plugin_names;

  // 我们将从ROS参数服务器获取要加载的规划器名称，然后加载规划器，并确保捕获所有异常。
  if (!motion_planning_api_tutorial_node->get_parameter("ompl.planning_plugins", planner_plugin_names))
    RCLCPP_FATAL(LOGGER, "Could not find planner plugin names");
  try
  {
    planner_plugin_loader.reset(new pluginlib::ClassLoader<planning_interface::PlannerManager>(
        "moveit_core", "planning_interface::PlannerManager"));
  }
  catch (pluginlib::PluginlibException& ex)
  {
    RCLCPP_FATAL(LOGGER, "Exception while creating planning plugin loader %s", ex.what());
  }

  if (planner_plugin_names.empty())
  {
    RCLCPP_ERROR(LOGGER,
                 "No planner plugins defined. Please make sure that the planning_plugins parameter is not empty.");
    return -1;
  }

  const auto& planner_name = planner_plugin_names.at(0);
  try
  {
    planner_instance.reset(planner_plugin_loader->createUnmanagedInstance(planner_name));
    if (!planner_instance->initialize(robot_model, motion_planning_api_tutorial_node,
                                      motion_planning_api_tutorial_node->get_namespace()))
      RCLCPP_FATAL(LOGGER, "Could not initialize planner instance");
    RCLCPP_INFO(LOGGER, "Using planning interface '%s'", planner_instance->getDescription().c_str());
  }
  catch (pluginlib::PluginlibException& ex)
  {
    const std::vector<std::string>& classes = planner_plugin_loader->getDeclaredClasses();
    std::stringstream ss;
    for (const auto& cls : classes)
      ss << cls << " ";
    RCLCPP_ERROR(LOGGER, "Exception while loading planner '%s': %s\nAvailable plugins: %s", planner_name.c_str(),
                 ex.what(), ss.str().c_str());
  }

  moveit::planning_interface::MoveGroupInterface move_group(motion_planning_api_tutorial_node, PLANNING_GROUP);

  // Visualization
  // ^^^^^^^^^^^^^
  // MoveItVisualTools软件包提供了在RViz中可视化对象、机器人和轨迹等多种功能，以及调试工具，如逐步检查脚本。
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(motion_planning_api_tutorial_node, "panda_link0",
                                                      "move_group_tutorial", move_group.getRobotModel());
  visual_tools.enableBatchPublishing();
  visual_tools.deleteAllMarkers();  // clear all old markers
  visual_tools.trigger();

  // 远程控制是一种内省工具，允许用户通过RViz中的按钮和键盘快捷键逐步执行高级脚本
  visual_tools.loadRemoteControl();

  // RViz提供了许多类型的标记，在这个演示中，我们将使用文本、圆柱体和球体
  Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
  text_pose.translation().z() = 1.75;
  visual_tools.publishText(text_pose, "Motion Planning API Demo", rvt::WHITE, rvt::XLARGE);

  // 批处理发布用于减少发送到RViz的大型可视化消息数量
  visual_tools.trigger();

  // 我们还可以使用visual_tools等待用户输入
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  // Pose Goal
  // 位姿目标
  // ^^^^^^^^^
  // 我们将为Panda的手臂创建一个运动计划请求，指定末端执行器的期望姿态作为输入。
  visual_tools.trigger();
  planning_interface::MotionPlanRequest req;
  planning_interface::MotionPlanResponse res;
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "panda_link0";
  pose.pose.position.x = 0.3;
  pose.pose.position.y = 0.4;
  pose.pose.position.z = 0.75;
  pose.pose.orientation.w = 1.0;

  // 在位置上指定了0.01米的公差，在方向上指定了0.01弧度的公差
  std::vector<double> tolerance_pose(3, 0.01);
  std::vector<double> tolerance_angle(3, 0.01);

  // 我们将使用来自kinematic_constraints软件包的可用辅助函数将请求创建为约束。
  moveit_msgs::msg::Constraints pose_goal = kinematic_constraints::constructGoalConstraints("panda_link8", pose, tolerance_pose, tolerance_angle);

  req.group_name = PLANNING_GROUP;
  req.goal_constraints.push_back(pose_goal);

  // 定义工作空间边界
  req.workspace_parameters.min_corner.x = req.workspace_parameters.min_corner.y = req.workspace_parameters.min_corner.z = -5.0;
  req.workspace_parameters.max_corner.x = req.workspace_parameters.max_corner.y = req.workspace_parameters.max_corner.z = 5.0;

  // 我们现在构建一个封装场景、请求和响应的计划上下文。我们使用这个计划上下文调用计划器
  planning_interface::PlanningContextPtr context = planner_instance->getPlanningContext(planning_scene, req, res.error_code);

  if (!context)
  {
    RCLCPP_ERROR(LOGGER, "Failed to create planning context");
    return -1;
  }
  context->solve(res);
  if (res.error_code.val != res.error_code.SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "Could not compute plan successfully");
    return -1;
  }

  // Visualize the result
  // 可视化结果
  // ^^^^^^^^^^^^^^^^^^^^
  std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
      motion_planning_api_tutorial_node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path",
                                                                                               1);
  moveit_msgs::msg::DisplayTrajectory display_trajectory;

  /* Visualize the trajectory */
  moveit_msgs::msg::MotionPlanResponse response;
  res.getMessage(response);

  display_trajectory.trajectory_start = response.trajectory_start;
  display_trajectory.trajectory.push_back(response.trajectory);
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();
  display_publisher->publish(display_trajectory);

  // 将计划场景中的状态设置为最后一个计划的最终状态
  robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
  planning_scene->setCurrentState(*robot_state.get());

  // Display the goal state
  // 显示目标状态
  visual_tools.publishAxisLabeled(pose.pose, "goal_1");
  visual_tools.publishText(text_pose, "Pose Goal (1)", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  // 我们也可以使用visual_tools等待用户输入
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Joint Space Goals
  // 关节空间目标
  // ^^^^^^^^^^^^^^^^^
  // 现在，设置一个关节空间目标
  moveit::core::RobotState goal_state(robot_model);
  std::vector<double> joint_values = { -1.0, 0.7, 0.7, -1.5, -0.7, 2.0, 0.0 };
  goal_state.setJointGroupPositions(joint_model_group, joint_values);
  moveit_msgs::msg::Constraints joint_goal = kinematic_constraints::constructGoalConstraints(goal_state, joint_model_group);
  req.goal_constraints.clear();
  req.goal_constraints.push_back(joint_goal);

  // 调用规划器并可视化轨迹 重新构建规划上下文
  context = planner_instance->getPlanningContext(planning_scene, req, res.error_code);
  /* Call the Planner */
  context->solve(res);
  /* Check that the planning was successful */
  if (res.error_code.val != res.error_code.SUCCESS)
  {
    RCLCPP_ERROR(LOGGER, "Could not compute plan successfully");
    return -1;
  }
  /* Visualize the trajectory */
  res.getMessage(response);
  display_trajectory.trajectory.push_back(response.trajectory);

  /* Now you should see two planned trajectories in series*/
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();
  display_publisher->publish(display_trajectory);

  /* We will add more goals. But first, set the state in the planning scene to the final state of the last plan */
  robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
  planning_scene->setCurrentState(*robot_state.get());

  // Display the goal state
  visual_tools.publishAxisLabeled(pose.pose, "goal_2");
  visual_tools.publishText(text_pose, "Joint Space Goal (2)", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  /* Wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // 回到第一个目标，准备进行方向约束规划
  req.goal_constraints.clear();
  req.goal_constraints.push_back(pose_goal);
  context = planner_instance->getPlanningContext(planning_scene, req, res.error_code);
  context->solve(res);
  res.getMessage(response);

  display_trajectory.trajectory.push_back(response.trajectory);
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();
  display_publisher->publish(display_trajectory);

  /* Set the state in the planning scene to the final state of the last plan */
  robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
  planning_scene->setCurrentState(*robot_state.get());

  // Display the goal state
  visual_tools.trigger();

  /* Wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Adding Path Constraints
  // 添加路径约束
  // ^^^^^^^^^^^^^^^^^^^^^^^
  // 翻译：再次添加一个新的姿态目标。这次我们还将添加一个路径约束到运动中。
  /* Let's create a new pose goal */

  pose.pose.position.x = 0.32;
  pose.pose.position.y = -0.25;
  pose.pose.position.z = 0.65;
  pose.pose.orientation.w = 1.0;
  moveit_msgs::msg::Constraints pose_goal_2 = kinematic_constraints::constructGoalConstraints("panda_link8", pose, tolerance_pose, tolerance_angle);

  /* Now, let's try to move to this new pose goal*/
  req.goal_constraints.clear();
  req.goal_constraints.push_back(pose_goal_2);

  // 让我们在运动中施加一个路径约束。在这里，我们要求末端执行器保持水平
  geometry_msgs::msg::QuaternionStamped quaternion;
  quaternion.header.frame_id = "panda_link0";
  req.path_constraints = kinematic_constraints::constructGoalConstraints("panda_link8", quaternion);

  // 施加路径约束需要规划器在末端执行器可能的位置空间中推理（机器人的工作空间）
  // 因为规划器需要在末端执行器可能的位置空间中推理（机器人的工作空间），所以我们需要指定一个允许规划体积的边界
  // 注意：默认边界由WorkspaceBounds请求适配器自动填充（OMPL管道的一部分，但在本示例中未使用）。
  // 我们使用一个肯定包括手臂可达空间的边界。这是可以的，因为规划手臂时不会在这个体积中进行采样；边界只用于确定采样配置是否有效。
  req.workspace_parameters.min_corner.x = req.workspace_parameters.min_corner.y =
      req.workspace_parameters.min_corner.z = -2.0;
  req.workspace_parameters.max_corner.x = req.workspace_parameters.max_corner.y =
      req.workspace_parameters.max_corner.z = 2.0;

  // 调用规划器并可视化到目前为止创建的所有计划。
  context = planner_instance->getPlanningContext(planning_scene, req, res.error_code);
  context->solve(res);
  res.getMessage(response);
  display_trajectory.trajectory.push_back(response.trajectory);
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();
  display_publisher->publish(display_trajectory);

  /* Set the state in the planning scene to the final state of the last plan */
  robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
  planning_scene->setCurrentState(*robot_state.get());

  // Display the goal state
  visual_tools.publishAxisLabeled(pose.pose, "goal_3");
  visual_tools.publishText(text_pose, "Orientation Constrained Motion Plan (3)", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  // END_TUTORIAL
  /* Wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to exit the demo");
  planner_instance.reset();

  rclcpp::shutdown();
  return 0;
}
```

## 启动文件

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    return LaunchDescription(
        [
            Node(
                package="moveit2_tutorials",
                executable="motion_planning_api_tutorial",
                name="motion_planning_api_tutorial",
                output="screen",
                parameters=[moveit_config.to_dict()],
            )
        ]
    )
```
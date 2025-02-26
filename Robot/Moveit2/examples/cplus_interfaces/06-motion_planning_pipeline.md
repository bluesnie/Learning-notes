###### datetime:2025/01/27 11:26

###### author:nzb

# [运动规划管道流](https://moveit.picknik.ai/main/doc/examples/motion_planning_pipeline/motion_planning_pipeline_tutorial.html#motion-planning-pipeline)

在 MoveIt 中，运动规划器被设置为规划路径。 然而，有时我们可能需要对运动规划请求进行预处理或对规划路径进行后处理（例如，时间参数化）。 
在这种情况下，我们使用规划流水线，将运动规划器与预处理和后处理阶段串联起来。 前处理和后处理阶段称为规划请求适配器，可通过 ROS 参数服务器按名称进行配置。 
在本教程中，我们将通过 C++ 代码来实例化和调用这样一个规划流水线。

## 功能目录
- 运动规划可视化
- 设置运动规划目标位姿
- 可视化结果
- 关节空间目标
- 规划请求适配器

## 代码解读

```c++
#include <pluginlib/class_loader.hpp>

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/planning_pipeline/planning_pipeline.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_visual_tools/moveit_visual_tools.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("motion_planning_pipeline");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("motion_planning_pipeline_tutorial", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // BEGIN_TUTORIAL
  // Start
  // ^^^^^
  // 开始使用规划管道非常容易。在我们加载规划器之前，我们需要两个对象，一个RobotModel和一个PlanningScene。
  // 我们将首先实例化一个RobotModelLoader对象，它将在ROS参数服务器上查找机器人描述，并为我们构建一个RobotModel。
  robot_model_loader::RobotModelLoaderPtr robot_model_loader(new robot_model_loader::RobotModelLoader(node, "robot_description"));

  // 使用RobotModelLoader，我们可以构建一个规划场景监视器，它将创建一个规划场景，监视规划场景差异，并将其应用于其内部规划场景。
  // 然后我们调用startSceneMonitor，startWorldGeometryMonitor和startStateMonitor来完全初始化规划场景监视器
  planning_scene_monitor::PlanningSceneMonitorPtr psm(
      new planning_scene_monitor::PlanningSceneMonitor(node, robot_model_loader));

  // 监听/XXX主题上的规划场景消息，并将其应用于内部规划场景
  psm->startSceneMonitor();
  // 监听世界几何、碰撞对象和（可选）八叉树世界几何、碰撞对象和（可选）八叉树的变化
  psm->startWorldGeometryMonitor();
  // 监听关节状态更新以及附加碰撞对象的变化，并相应地更新内部规划场景
  psm->startStateMonitor();

  // 我们还可以使用RobotModelLoader来获取包含机器人运动学信息的机器人模型
  moveit::core::RobotModelPtr robot_model = robot_model_loader->getModel();

  // 我们可以通过锁定内部规划场景以进行读取来从PlanningSceneMonitor获取最新的机器人状态。这个锁确保我们在读取其状态时底层场景不会被更新。 
  // RobotState对于计算机器人的正向和反向运动学以及许多其他用途非常有用。
  moveit::core::RobotStatePtr robot_state(
      new moveit::core::RobotState(planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentState()));

  // 创建一个JointModelGroup来跟踪当前机器人的姿态和规划组。Joint Model group对于一次处理一组关节非常有用，例如左臂或末端执行器
  const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup("panda_arm");

  // 我们现在可以设置PlanningPipeline对象，它将使用ROS参数服务器来确定要使用的请求适配器和规划插件集
  planning_pipeline::PlanningPipelinePtr planning_pipeline(
      new planning_pipeline::PlanningPipeline(robot_model, node, "ompl"));

  // Visualization
  // ^^^^^^^^^^^^^
  // The package MoveItVisualTools provides many capabilities for visualizing objects, robots,
  // and trajectories in RViz as well as debugging tools such as step-by-step introspection of a script.
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "panda_link0", "move_group_tutorial", psm);
  visual_tools.deleteAllMarkers();

  /* Remote control is an introspection tool that allows users to step through a high level script
     via buttons and keyboard shortcuts in RViz */
  visual_tools.loadRemoteControl();

  /* RViz provides many types of markers, in this demo we will use text, cylinders, and spheres*/
  Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
  text_pose.translation().z() = 1.75;
  visual_tools.publishText(text_pose, "Motion Planning Pipeline Demo", rvt::WHITE, rvt::XLARGE);

  /* Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations */
  visual_tools.trigger();

  /* We can also use visual_tools to wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  // Pose Goal
  // ^^^^^^^^^
  // We will now create a motion plan request for the right arm of the Panda
  // specifying the desired pose of the end-effector as input.
  planning_interface::MotionPlanRequest req;
  req.pipeline_id = "ompl";
  req.planner_id = "RRTConnectkConfigDefault";
  req.allowed_planning_time = 1.0;
  req.max_velocity_scaling_factor = 1.0;
  req.max_acceleration_scaling_factor = 1.0;
  planning_interface::MotionPlanResponse res;
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "panda_link0";
  pose.pose.position.x = 0.3;
  pose.pose.position.y = 0.0;
  pose.pose.position.z = 0.75;
  pose.pose.orientation.w = 1.0;

  // A tolerance of 0.01 m is specified in position
  // and 0.01 radians in orientation
  std::vector<double> tolerance_pose(3, 0.1);
  std::vector<double> tolerance_angle(3, 0.1);

  // We will create the request as a constraint using a helper
  // function available from the
  // :moveit_codedir:`kinematic_constraints<moveit_core/kinematic_constraints/include/moveit/kinematic_constraints/kinematic_constraint.hpp>`
  // package.
  req.group_name = "panda_arm";
  moveit_msgs::msg::Constraints pose_goal =
      kinematic_constraints::constructGoalConstraints("panda_link8", pose, tolerance_pose, tolerance_angle);
  req.goal_constraints.push_back(pose_goal);

  // Before planning, we will need a Read Only lock on the planning scene so that it does not modify the world
  // representation while planning
  {
    planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
    /* Now, call the pipeline and check whether planning was successful. */
    /* Check that the planning was successful */
    if (!planning_pipeline->generatePlan(lscene, req, res) || res.error_code.val != res.error_code.SUCCESS)
    {
      RCLCPP_ERROR(LOGGER, "Could not compute plan successfully");
      rclcpp::shutdown();
      return -1;
    }
  }
  // Visualize the result
  // ^^^^^^^^^^^^^^^^^^^^
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_publisher =
      node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
  moveit_msgs::msg::DisplayTrajectory display_trajectory;

  /* Visualize the trajectory */
  RCLCPP_INFO(LOGGER, "Visualizing the trajectory");
  moveit_msgs::msg::MotionPlanResponse response;
  res.getMessage(response);

  display_trajectory.trajectory_start = response.trajectory_start;
  display_trajectory.trajectory.push_back(response.trajectory);
  display_publisher->publish(display_trajectory);
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();

  /* Wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Joint Space Goals
  // ^^^^^^^^^^^^^^^^^
  /* First, set the state in the planning scene to the final state of the last plan */
  robot_state = planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentStateUpdated(response.trajectory_start);
  robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
  moveit::core::robotStateToRobotStateMsg(*robot_state, req.start_state);

  // Now, setup a joint space goal
  moveit::core::RobotState goal_state(*robot_state);
  std::vector<double> joint_values = { -1.0, 0.7, 0.7, -1.5, -0.7, 2.0, 0.0 };
  goal_state.setJointGroupPositions(joint_model_group, joint_values);
  moveit_msgs::msg::Constraints joint_goal =
      kinematic_constraints::constructGoalConstraints(goal_state, joint_model_group);

  req.goal_constraints.clear();
  req.goal_constraints.push_back(joint_goal);

  // Before planning, we will need a Read Only lock on the planning scene so that it does not modify the world
  // representation while planning
  {
    planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
    /* Now, call the pipeline and check whether planning was successful. */
    if (!planning_pipeline->generatePlan(lscene, req, res) || res.error_code.val != res.error_code.SUCCESS)
    {
      RCLCPP_ERROR(LOGGER, "Could not compute plan successfully");
      rclcpp::shutdown();
      return -1;
    }
  }
  /* Visualize the trajectory */
  RCLCPP_INFO(LOGGER, "Visualizing the trajectory");
  res.getMessage(response);
  display_trajectory.trajectory_start = response.trajectory_start;
  display_trajectory.trajectory.push_back(response.trajectory);
  // Now you should see two planned trajectories in series
  display_publisher->publish(display_trajectory);
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();

  /* Wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Using a Planning Request Adapter
  // 使用规划请求适配器
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 规划请求适配器允许我们指定一系列应在规划开始之前或规划完成后的路径上执行的操作

  /* First, set the state in the planning scene to the final state of the last plan */
  robot_state = planning_scene_monitor::LockedPlanningSceneRO(psm)->getCurrentStateUpdated(response.trajectory_start);
  robot_state->setJointGroupPositions(joint_model_group, response.trajectory.joint_trajectory.points.back().positions);
  moveit::core::robotStateToRobotStateMsg(*robot_state, req.start_state);

  // 现在，将其中一个关节稍微设置在它的上限之外
  const moveit::core::JointModel* joint_model = joint_model_group->getJointModel("panda_joint3");
  const moveit::core::JointModel::Bounds& joint_bounds = joint_model->getVariableBounds();
  std::vector<double> tmp_values(1, 0.0);
  tmp_values[0] = joint_bounds[0].min_position_ - 0.01;
  robot_state->setJointPositions(joint_model, tmp_values);

  req.goal_constraints.clear();
  req.goal_constraints.push_back(pose_goal);

  // Before planning, we will need a Read Only lock on the planning scene so that it does not modify the world
  // representation while planning
  {
    planning_scene_monitor::LockedPlanningSceneRO lscene(psm);
    /* Now, call the pipeline and check whether planning was successful. */
    if (!planning_pipeline->generatePlan(lscene, req, res) || res.error_code.val != res.error_code.SUCCESS)
    {
      RCLCPP_ERROR(LOGGER, "Could not compute plan successfully");
      rclcpp::shutdown();
      return -1;
    }
  }
  /* Visualize the trajectory */
  RCLCPP_INFO(LOGGER, "Visualizing the trajectory");
  res.getMessage(response);
  display_trajectory.trajectory_start = response.trajectory_start;
  display_trajectory.trajectory.push_back(response.trajectory);
  /* Now you should see three planned trajectories in series*/
  display_publisher->publish(display_trajectory);
  visual_tools.publishTrajectoryLine(display_trajectory.trajectory.back(), joint_model_group);
  visual_tools.trigger();

  /* Wait for user input */
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to finish the demo");

  RCLCPP_INFO(LOGGER, "Done");

  rclcpp::shutdown();
  return 0;
}
```

## 启动文件

```python
import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return file.read()
    # parent of IOError, OSError *and* WindowsError where available
    except EnvironmentError:
        return None


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    # parent of IOError, OSError *and* WindowsError where available
    except EnvironmentError:
        return None


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .planning_pipelines("ompl", ["ompl"])
        .to_moveit_configs()
    )

    # MotionPlanningPipeline demo executable
    motion_planning_pipeline_demo = Node(
        name="motion_planning_pipeline_tutorial",
        package="moveit2_tutorials",
        executable="motion_planning_pipeline_tutorial",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([motion_planning_pipeline_demo])
```
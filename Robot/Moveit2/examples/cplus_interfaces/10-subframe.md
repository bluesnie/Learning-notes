###### datetime:2025/02/13 17:39

###### author:nzb

# [子帧](https://moveit.picknik.ai/main/doc/examples/subframes/subframes_tutorial.html#subframes)


子帧是在碰撞对象上定义的帧。 它们可用于在场景中放置的物体上定义兴趣点，例如瓶口、螺丝刀尖或螺丝头。 它们可用于规划和编写机器人指令，如 "拿起瓶子，然后将瓶口移到水龙头喷嘴下方"，或 "拿起螺丝刀，然后将其放在螺丝钉头的上方"。 

以机器人操纵的对象为中心编写代码，不仅更具可读性，而且更稳健，机器人之间也更容易移植。 本教程将向您展示如何在碰撞对象上定义子帧，并将其发布到规划场景中，然后使用它们来规划运动。

## 功能目录
  - 使用子帧定义两个碰撞对象
  - 创建规划请求
  - 准备场景
  - 交互式测试机器人

## 代码解读

```cpp
// ROS
#include <rclcpp/rclcpp.hpp>

// MoveIt
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/move_group_interface/move_group_interface.h>

// TF2
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("subframes_tutorial");

// BEGIN_SUB_TUTORIAL plan1
//
// Creating the planning request
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// 在这个教程中，我们使用一个小助手函数来创建我们的规划请求并移动机器人。
bool moveToCartPose(const geometry_msgs::msg::PoseStamped &pose, moveit::planning_interface::MoveGroupInterface &group,
                    const std::string &end_effector_link)
{
  // 要在规划中使用附加到机器人的对象的子帧，您需要将move_group的末端执行器设置为对象的子帧。格式必须是“对象名称/子帧名称”，如“Example 1”行所示。
  // Do not forget to reset your end_effector_link to a robot link when you detach your object, and the subframe is not part of your robot anymore!
  // 不要忘记在分离对象时将move_group的末端执行器重置为机器人链接，并且子帧不再是机器人的一部分！
  group.clearPoseTargets();
  group.setEndEffectorLink(end_effector_link);
  /*
  group.setEndEffectorLink("cylinder/tip");    // Example 1
  group.setEndEffectorLink("panda_hand");      // Example 2
  */
  group.setStartStateToCurrentState();
  group.setPoseTarget(pose);

  // The rest of the planning is done as usual. Naturally, you can also use the ``go()`` command instead of
  // ``plan()`` and ``execute()``.
  RCLCPP_INFO_STREAM(LOGGER, "Planning motion to pose:");
  RCLCPP_INFO_STREAM(LOGGER, pose.pose.position.x << ", " << pose.pose.position.y << ", " << pose.pose.position.z);
  moveit::planning_interface::MoveGroupInterface::Plan myplan;
  if (group.plan(myplan) && group.execute(myplan))
    return true;

  RCLCPP_WARN(LOGGER, "Failed to perform motion.");
  return false;
}
// END_SUB_TUTORIAL

// BEGIN_SUB_TUTORIAL object1
//
// Defining two CollisionObjects with subframes
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// 这个辅助函数创建了两个对象并将它们发布到PlanningScene：一个盒子和一个圆柱体。盒子出现在夹爪前面，圆柱体出现在夹爪尖端，就像它被夹住了一样。

void spawnCollisionObjects(moveit::planning_interface::PlanningSceneInterface &planning_scene_interface)
{
  double z_offset_box = .25; // The z-axis points away from the gripper
  double z_offset_cylinder = .12;

  // First, we start defining the `CollisionObject <http://docs.ros.org/api/moveit_msgs/html/msg/CollisionObject.html>`_
  // as usual.
  moveit_msgs::msg::CollisionObject box;
  box.id = "box";
  box.header.frame_id = "panda_hand";
  box.primitives.resize(1);
  box.primitive_poses.resize(1);
  box.primitives[0].type = box.primitives[0].BOX;
  box.primitives[0].dimensions.resize(3);
  box.primitives[0].dimensions[0] = 0.05;
  box.primitives[0].dimensions[1] = 0.1;
  box.primitives[0].dimensions[2] = 0.02;
  box.primitive_poses[0].position.z = z_offset_box;

  // 然后，我们定义CollisionObject的子帧。子帧在``frame_id``坐标系中定义，就像组成对象的形状一样。每个子帧由一个名称和一个姿态组成。
  // 在这个教程中，我们设置了子帧的朝向，使子帧的z轴指向远离对象的方向。
  // 这并不严格必要，但遵循一个约定是有帮助的，并且在稍后设置目标姿态的朝向时可以避免混淆。
  box.subframe_names.resize(5);
  box.subframe_poses.resize(5);

  box.subframe_names[0] = "bottom";
  box.subframe_poses[0].position.y = 0.05; // -.05;
  box.subframe_poses[0].position.z = 0.0 + z_offset_box;

  tf2::Quaternion orientation;
  orientation.setRPY(90.0 / 180.0 * M_PI, 0, 0);
  box.subframe_poses[0].orientation = tf2::toMsg(orientation);
  // END_SUB_TUTORIAL

  box.subframe_names[1] = "top";
  box.subframe_poses[1].position.y = -0.05; // .05;
  box.subframe_poses[1].position.z = 0.05;  // 0.0 + z_offset_box;
  orientation.setRPY(-90.0 / 180.0 * M_PI, 0, 0);
  box.subframe_poses[1].orientation = tf2::toMsg(orientation);

  box.subframe_names[2] = "corner_1";
  box.subframe_poses[2].position.x = -.025;
  box.subframe_poses[2].position.y = -.05;
  box.subframe_poses[2].position.z = -.01 + z_offset_box;
  orientation.setRPY(90.0 / 180.0 * M_PI, 0, 0);
  box.subframe_poses[2].orientation = tf2::toMsg(orientation);

  box.subframe_names[3] = "corner_2";
  box.subframe_poses[3].position.x = .025;
  box.subframe_poses[3].position.y = -.05;
  box.subframe_poses[3].position.z = -.01 + z_offset_box;
  orientation.setRPY(90.0 / 180.0 * M_PI, 0, 0);
  box.subframe_poses[3].orientation = tf2::toMsg(orientation);

  box.subframe_names[4] = "side";
  box.subframe_poses[4].position.x = .0;
  box.subframe_poses[4].position.y = .0;
  box.subframe_poses[4].position.z = 0.0; // -.01 + z_offset_box;
  orientation.setRPY(0, 180.0 / 180.0 * M_PI, 0);
  box.subframe_poses[4].orientation = tf2::toMsg(orientation);

  // Next, define the cylinder
  moveit_msgs::msg::CollisionObject cylinder;
  cylinder.id = "cylinder";
  cylinder.header.frame_id = "panda_hand";
  cylinder.primitives.resize(1);
  cylinder.primitive_poses.resize(1);
  cylinder.primitives[0].type = box.primitives[0].CYLINDER;
  cylinder.primitives[0].dimensions.resize(2);
  cylinder.primitives[0].dimensions[0] = 0.06;  // height (along x)
  cylinder.primitives[0].dimensions[1] = 0.005; // radius
  cylinder.primitive_poses[0].position.x = 0.0;
  cylinder.primitive_poses[0].position.y = 0.0;
  cylinder.primitive_poses[0].position.z = 0.0 + z_offset_cylinder;
  orientation.setRPY(0, 90.0 / 180.0 * M_PI, 0);
  cylinder.primitive_poses[0].orientation = tf2::toMsg(orientation);

  cylinder.subframe_poses.resize(1);
  cylinder.subframe_names.resize(1);
  cylinder.subframe_names[0] = "tip";
  cylinder.subframe_poses[0].position.x = 0.03;
  cylinder.subframe_poses[0].position.y = 0.0;
  cylinder.subframe_poses[0].position.z = 0.0 + z_offset_cylinder;
  orientation.setRPY(0, 90.0 / 180.0 * M_PI, 0);
  cylinder.subframe_poses[0].orientation = tf2::toMsg(orientation);

  // BEGIN_SUB_TUTORIAL object2
  // 最后，将对象发布到PlanningScene。在本教程中，我们发布一个立方体和一个圆柱体。
  box.operation = moveit_msgs::msg::CollisionObject::ADD;
  cylinder.operation = moveit_msgs::msg::CollisionObject::ADD;
  planning_scene_interface.applyCollisionObjects({box, cylinder});
}
// END_SUB_TUTORIAL

void createArrowMarker(visualization_msgs::msg::Marker &marker, const geometry_msgs::msg::Pose &pose, const Eigen::Vector3d &dir,
                       int id, double scale = 0.1)
{
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.type = visualization_msgs::msg::Marker::CYLINDER;
  marker.id = id;
  marker.scale.x = 0.1 * scale;
  marker.scale.y = 0.1 * scale;
  marker.scale.z = scale;

  Eigen::Isometry3d pose_eigen;
  tf2::fromMsg(pose, pose_eigen);
  marker.pose = tf2::toMsg(pose_eigen * Eigen::Translation3d(dir * (0.5 * scale)) *
                           Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), dir));

  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;
}

void createFrameMarkers(visualization_msgs::msg::MarkerArray &markers, const geometry_msgs::msg::PoseStamped &target,
                        const std::string &ns, bool locked = false)
{
  int id = markers.markers.size();
  visualization_msgs::msg::Marker m;
  m.header.frame_id = target.header.frame_id;
  m.ns = ns;
  m.frame_locked = locked;

  createArrowMarker(m, target.pose, Eigen::Vector3d::UnitX(), ++id);
  m.color.r = 1.0;
  markers.markers.push_back(m);
  createArrowMarker(m, target.pose, Eigen::Vector3d::UnitY(), ++id);
  m.color.g = 1.0;
  markers.markers.push_back(m);
  createArrowMarker(m, target.pose, Eigen::Vector3d::UnitZ(), ++id);
  m.color.b = 1.0;
  markers.markers.push_back(m);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("subframes_tutorial");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]()
                             { executor.spin(); });
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  moveit::planning_interface::MoveGroupInterface group(node, "panda_arm");
  group.setPlanningTime(60.0);

  // moveit::core::RobotState start_state(*group.getCurrentState());
  // group.setStartState(start_state);

  // BEGIN_SUB_TUTORIAL sceneprep
  // Preparing the scene
  // ^^^^^^^^^^^^^^^^^^^
  // 在主函数中，我们首先在规划场景中生成对象，然后将圆柱体连接到机器人上。
  // 将圆柱体连接到机器人上，使其在Rviz中变为紫色。
  spawnCollisionObjects(planning_scene_interface);
  moveit_msgs::msg::AttachedCollisionObject att_coll_object;
  att_coll_object.object.id = "cylinder";
  att_coll_object.link_name = "panda_hand";
  att_coll_object.object.operation = att_coll_object.object.ADD;
  RCLCPP_INFO_STREAM(LOGGER, "Attaching cylinder to robot.");
  planning_scene_interface.applyAttachedCollisionObject(att_coll_object);
  // END_SUB_TUTORIAL
  // 获取当前规划场景状态一次
  auto planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node, "robot_description");
  planning_scene_monitor->requestPlanningSceneState();
  planning_scene_monitor::LockedPlanningSceneRO planning_scene(planning_scene_monitor);

  // Visualize frames as rviz markers
  auto marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_array", 10);
  auto showFrames = [&](geometry_msgs::msg::PoseStamped target, const std::string &eef)
  {
    visualization_msgs::msg::MarkerArray markers;
    // 将目标姿态转换为规划框架位姿
    Eigen::Isometry3d tf;
    tf2::fromMsg(target.pose, tf);
    target.pose = tf2::toMsg(planning_scene->getFrameTransform(target.header.frame_id) * tf);
    target.header.frame_id = planning_scene->getPlanningFrame();
    createFrameMarkers(markers, target, "target");

    // 将eef转换为相对于panda_hand的位姿
    target.header.frame_id = "panda_hand";
    target.pose = tf2::toMsg(planning_scene->getFrameTransform(target.header.frame_id).inverse() *
                             planning_scene->getFrameTransform(eef));
    createFrameMarkers(markers, target, "eef", true);

    marker_publisher->publish(markers);
  };

  // 定义一个在机器人基座上的姿态。
  tf2::Quaternion target_orientation;
  geometry_msgs::msg::PoseStamped fixed_pose, target_pose;
  fixed_pose.header.frame_id = "panda_link0";
  fixed_pose.pose.position.y = -.4;
  fixed_pose.pose.position.z = .3;
  target_orientation.setRPY(0, (-20.0 / 180.0 * M_PI), 0);
  fixed_pose.pose.orientation = tf2::toMsg(target_orientation);

  // 设置一个小型命令行界面，使教程具有交互性。
  int character_input;
  while (rclcpp::ok())
  {
    RCLCPP_INFO(LOGGER, "==========================\n"
                        "Press a key and hit Enter to execute an action. \n0 to exit"
                        "\n1 to move cylinder tip to box bottom \n2 to move cylinder tip to box top"
                        "\n3 to move cylinder tip to box corner 1 \n4 to move cylinder tip to box corner 2"
                        "\n5 to move cylinder tip to side of box"
                        "\n6 to return the robot to the start pose"
                        "\n7 to move the robot's wrist to a cartesian pose"
                        "\n8 to move cylinder/tip to the same cartesian pose"
                        "\n----------"
                        "\n10 to remove box and cylinder from the scene"
                        "\n11 to spawn box and cylinder"
                        "\n12 to attach the cylinder to the gripper\n");
    std::cin >> character_input;
    if (character_input == 0)
    {
      return 0;
    }
    else if (character_input == 1)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Moving to bottom of box with cylinder tip");

      // BEGIN_SUB_TUTORIAL orientation
      // Setting the orientation
      // ^^^^^^^^^^^^^^^^^^^^^^^
      // The target pose is given relative to a box subframe:
      // 设置目标姿态，相对于一个box子框架：
      target_pose.header.frame_id = "box/bottom";
      // The orientation is determined by RPY angles to align the cylinder and box subframes:
      // 通过RPY角度确定姿态，使圆柱体和box子框架对齐：
      target_orientation.setRPY(0, 180.0 / 180.0 * M_PI, 90.0 / 180.0 * M_PI);
      target_pose.pose.orientation = tf2::toMsg(target_orientation);
      // To keep some distance to the box, we use a small offset:
      // 为了保持与box一定的距离，我们使用一个小偏移量：
      target_pose.pose.position.z = 0.01;
      showFrames(target_pose, "cylinder/tip");
      moveToCartPose(target_pose, group, "cylinder/tip");
      // END_SUB_TUTORIAL
    }
    // BEGIN_SUB_TUTORIAL move_example
    // 命令“2”将圆柱体尖端移动到box的顶部
    else if (character_input == 2)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Moving to top of box with cylinder tip");
      target_pose.header.frame_id = "box/top";
      target_orientation.setRPY(180.0 / 180.0 * M_PI, 0, 90.0 / 180.0 * M_PI);
      target_pose.pose.orientation = tf2::toMsg(target_orientation);
      target_pose.pose.position.z = 0.01;
      showFrames(target_pose, "cylinder/tip");
      moveToCartPose(target_pose, group, "cylinder/tip");
    }
    // END_SUB_TUTORIAL
    else if (character_input == 3)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Moving to corner1 of box with cylinder tip");
      target_pose.header.frame_id = "box/corner_1";
      target_orientation.setRPY(0, 180.0 / 180.0 * M_PI, 90.0 / 180.0 * M_PI);
      target_pose.pose.orientation = tf2::toMsg(target_orientation);
      target_pose.pose.position.z = 0.01;
      showFrames(target_pose, "cylinder/tip");
      moveToCartPose(target_pose, group, "cylinder/tip");
    }
    else if (character_input == 4)
    {
      target_pose.header.frame_id = "box/corner_2";
      target_orientation.setRPY(0, 180.0 / 180.0 * M_PI, 90.0 / 180.0 * M_PI);
      target_pose.pose.orientation = tf2::toMsg(target_orientation);
      target_pose.pose.position.z = 0.01;
      showFrames(target_pose, "cylinder/tip");
      moveToCartPose(target_pose, group, "cylinder/tip");
    }
    else if (character_input == 5)
    {
      target_pose.header.frame_id = "box/side";
      target_orientation.setRPY(0, 180.0 / 180.0 * M_PI, 90.0 / 180.0 * M_PI);
      target_pose.pose.orientation = tf2::toMsg(target_orientation);
      target_pose.pose.position.z = 0.01;
      showFrames(target_pose, "cylinder/tip");
      moveToCartPose(target_pose, group, "cylinder/tip");
    }
    else if (character_input == 6)
    {
      // Go to neutral home pose
      group.clearPoseTargets();
      group.setNamedTarget("ready");
      group.move();
    }
    else if (character_input == 7)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Moving to a pose with robot wrist");
      showFrames(fixed_pose, "panda_hand");
      moveToCartPose(fixed_pose, group, "panda_hand");
    }
    else if (character_input == 8)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Moving to a pose with cylinder tip");
      showFrames(fixed_pose, "cylinder/tip");
      moveToCartPose(fixed_pose, group, "cylinder/tip");
    }
    else if (character_input == 10)
    {
      try
      {
        RCLCPP_INFO_STREAM(LOGGER, "Removing box and cylinder.");
        moveit_msgs::msg::AttachedCollisionObject att_coll_object;
        att_coll_object.object.id = "box";
        att_coll_object.object.operation = att_coll_object.object.REMOVE;
        planning_scene_interface.applyAttachedCollisionObject(att_coll_object);

        att_coll_object.object.id = "cylinder";
        att_coll_object.object.operation = att_coll_object.object.REMOVE;
        planning_scene_interface.applyAttachedCollisionObject(att_coll_object);

        moveit_msgs::msg::CollisionObject co1, co2;
        co1.id = "box";
        co1.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        co2.id = "cylinder";
        co2.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        planning_scene_interface.applyCollisionObjects({co1, co2});
      }
      catch (const std::exception &exc)
      {
        RCLCPP_WARN_STREAM(LOGGER, exc.what());
      }
    }
    else if (character_input == 11)
    {
      RCLCPP_INFO_STREAM(LOGGER, "Respawning test box and cylinder.");
      spawnCollisionObjects(planning_scene_interface);
    }
    else if (character_input == 12)
    {
      moveit_msgs::msg::AttachedCollisionObject att_coll_object;
      att_coll_object.object.id = "cylinder";
      att_coll_object.link_name = "panda_hand";
      att_coll_object.object.operation = att_coll_object.object.ADD;
      RCLCPP_INFO_STREAM(LOGGER, "Attaching cylinder to robot.");
      planning_scene_interface.applyAttachedCollisionObject(att_coll_object);
    }
    else
    {
      RCLCPP_INFO(LOGGER, "Could not read input. Quitting.");
      break;
    }
  }

  rclcpp::shutdown();
  return 0;
}

// BEGIN_TUTORIAL
// CALL_SUB_TUTORIAL object1
// CALL_SUB_TUTORIAL object2
// CALL_SUB_TUTORIAL plan1
//
// CALL_SUB_TUTORIAL sceneprep
//
// Interactively testing the robot
// 交互式测试机器人
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// We set up a small command line interface so you can interact with the simulation and see how it responds to certain commands.
// 我们设置了一个小命令行界面，以便您与模拟互动并查看它如何响应某些命令。
// You can use it to experiment with the behavior of the robot when you remove the box and cylinder, respawn and reattach them, or create new planning requests.
// 您可以使用它来实验机器人移除盒子、重新生成并重新附加它们或创建新的规划请求时的行为。
// Try moving the robot into a new position and respawn the box and cylinder there (they are spawned relative to the robot wrist).
// 尝试将机器人移动到新位置并在那里重新生成盒子和小球（它们是相对于机器人手腕生成的）。
// Or try commands 7 and 8 to move different frames to the same position in space.
// 或者尝试命令7和8将不同的框架移动到空间中的相同位置。
// CALL_SUB_TUTORIAL move_example
//
// CALL_SUB_TUTORIAL orientation
//
// END_TUTORIAL

``` 

## 技术说明

子帧不为 `TF` 所知，因此无法在 `MoveIt` 规划请求之外使用。 如果您需要子帧的变换，可以使用 `getFrameTransform` 函数从规划场景的碰撞机器人中获取。 这会返回一个 `Eigen::Isometry3d` 对象，从中可以提取平移和四元数（[请参阅此处](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html)）。 然后就可以使用平移和四元数创建变换，并将其注册到 `TFListener` 中。

## 启动文件

> 需要加一行

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # Command-line arguments
    tutorial_arg = DeclareLaunchArgument(
        "rviz_tutorial", default_value="False", description="Tutorial flag"
    )

    db_arg = DeclareLaunchArgument(
        "db", default_value="False", description="Database flag"
    )

    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="mock_components",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True   # 需要加这一行
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz
    tutorial_mode = LaunchConfiguration("rviz_tutorial")
    rviz_base = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"), "launch"
    )
    rviz_full_config = os.path.join(rviz_base, "moveit.rviz")
    rviz_empty_config = os.path.join(rviz_base, "moveit_empty.rviz")
    rviz_node_tutorial = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_empty_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        condition=IfCondition(tutorial_mode),
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_full_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        condition=UnlessCondition(tutorial_mode),
    )

    # Static TF
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )

    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
    )

    # Warehouse mongodb server
    db_config = LaunchConfiguration("db")
    mongodb_server_node = Node(
        package="warehouse_ros_mongo",
        executable="mongo_wrapper_ros.py",
        parameters=[
            {"warehouse_port": 33829},
            {"warehouse_host": "localhost"},
            {"warehouse_plugin": "warehouse_ros_mongo::MongoDatabaseConnection"},
        ],
        output="screen",
        condition=IfCondition(db_config),
    )

    return LaunchDescription(
        [
            tutorial_arg,
            db_arg,
            ros2_control_hardware_type,
            rviz_node,
            rviz_node_tutorial,
            static_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
            mongodb_server_node,
        ]
    )

```

## CMakelLists.txt

```
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)

ament_target_dependencies(
  subframes_tutorial
  "moveit_ros_planning_interface"
  "rclcpp"
  "tf2_geometry_msgs"
  "tf2_eigen"
)
```

 ## 运行

- `ros2 launch moveit_resources_panda_moveit_config demo.launch.py `
- 另一个终端 `ros2 run visualizing_collisions subframes_tutorial`
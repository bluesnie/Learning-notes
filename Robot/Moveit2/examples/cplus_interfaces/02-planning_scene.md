###### datetime:2025/01/24 14:15

###### author:nzb

# [规划场景](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html#planning-scene)

## 功能目录
  - 碰撞检测
    - 自碰撞检测
    - 修改状态
    - 以组进行碰撞检测，比如对`panda_arm`的`hand`组进行碰撞检测
    - 获取碰撞检测结果信息
    - 修改碰撞检测矩阵
    - 自身碰撞检测和环境碰撞检测
  - 约束检测
    - 运动学约束检测
      - 关节约束
      - 位置约束
      - 可见性约束
    - 用户自定义约束

## 代码解读

```c++
#include <rclcpp/rclcpp.hpp>

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/planning_scene/planning_scene.hpp>

#include <moveit/kinematic_constraints/utils.hpp>

// BEGIN_SUB_TUTORIAL stateFeasibilityTestExample
// 用户定义的约束也可以通过PlanningScene类指定。这是通过使用setStateFeasibilityPredicate函数指定回调来完成的。
// 这里有一个简单的示例，它使用setStateFeasibilityPredicate函数指定回调来检查Panda机器人的“panda_joint1”是否在正或负角度上：
bool stateFeasibilityTestExample(const moveit::core::RobotState& robot_state, bool /*verbose*/)
{
  const double* joint_values = robot_state.getJointPositions("panda_joint1");
  return (joint_values[0] > 0.0);
}
// END_SUB_TUTORIAL

static const rclcpp::Logger LOGGER = rclcpp::get_logger("planning_scene_tutorial");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto planning_scene_tutorial_node = rclcpp::Node::make_shared("planning_scene_tutorial", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(planning_scene_tutorial_node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // Setup
  // ^^^^^
  // PlanningScene类可以通过使用RobotModel或URDF和SRDF轻松设置和配置。然而，这不是实例化PlanningScene的推荐方法。
  // PlanningSceneMonitor是PlanningSceneMonitor的推荐方法，用于使用机器人关节和机器人上的传感器数据创建和维护当前规划场景（在下一个教程中详细讨论）。
  // 在教程中，我们将直接实例化一个PlanningScene类，但是这种实例化方法只用于说明。
  robot_model_loader::RobotModelLoader robot_model_loader(planning_scene_tutorial_node, "robot_description");
  const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScene planning_scene(kinematic_model);

  // Collision Checking
  // 碰撞检测
  // ^^^^^^^^^^^^^^^^^^
  // Self-collision checking
  // 自身碰撞检测
  // ~~~~~~~~~~~~~~~~~~~~~~~
  // 首先，我们将检查机器人是否处于自身碰撞状态，即机器人的当前配置是否会导致机器人的部分相互碰撞。
  // 为此，我们将构建一个CollisionRequest对象和一个CollisionResult对象，并将它们传递给碰撞检查函数。
  // 请注意，机器人是否处于自身碰撞状态的结果包含在结果中。自身碰撞检测使用的是*未填充*的机器人版本，即它直接使用URDF中提供的碰撞网格，没有添加额外的填充。

  collision_detection::CollisionRequest collision_request;
  collision_detection::CollisionResult collision_result;
  planning_scene.checkSelfCollision(collision_request, collision_result);
  RCLCPP_INFO_STREAM(LOGGER, "Test 1: Current state is " << (collision_result.collision ? "in" : "not in")
                                                         << " self collision");
  // Change the state
  // 更改状态
  // ~~~~~~~~~~~~~~~~
  // 现在，让我们改变机器人的当前状态。规划场景在内部维护当前状态。我们可以获取对它的引用并更改它，然后检查新机器人配置的碰撞。
  // 特别需要注意的是，在发出新的碰撞检查请求之前，我们需要清除collision_result。

  moveit::core::RobotState& current_state = planning_scene.getCurrentStateNonConst();
  current_state.setToRandomPositions();
  collision_result.clear();
  planning_scene.checkSelfCollision(collision_request, collision_result);
  RCLCPP_INFO_STREAM(LOGGER, "Test 2: Current state is " << (collision_result.collision ? "in" : "not in")
                                                         << " self collision");

  // Checking for a group
  // 检查组
  // ~~~~~~~~~~~~~~~~~~~~
  // 现在，我们将只检查Panda的手部是否与机器人身体的其它部分发生碰撞。我们可以通过将组名“hand”添加到碰撞请求中，具体要求只检查手部。

  collision_request.group_name = "hand";
  current_state.setToRandomPositions();
  collision_result.clear();
  planning_scene.checkSelfCollision(collision_request, collision_result);
  RCLCPP_INFO_STREAM(LOGGER, "Test 3: Current state is " << (collision_result.collision ? "in" : "not in")
                                                         << " self collision");

  // Getting Contact Information
  // 获取接触信息
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // 首先，手动将Panda手臂设置到一个我们知道会发生内部（自身）碰撞的位置。
  // 请注意，这个状态实际上超出了Panda的关节极限，我们也可以直接检查这一点。

  std::vector<double> joint_values = { 0.0, 0.0, 0.0, -2.9, 0.0, 1.4, 0.0 };
  const moveit::core::JointModelGroup* joint_model_group = current_state.getJointModelGroup("panda_arm");
  current_state.setJointGroupPositions(joint_model_group, joint_values);
  RCLCPP_INFO_STREAM(LOGGER, "Test 4: Current state is "
                                 << (current_state.satisfiesBounds(joint_model_group) ? "valid" : "not valid"));

  // 现在，我们可以获取Panda臂在给定配置下可能发生的任何碰撞的接触信息。我们可以通过在碰撞请求中填写适当的字段，并将要返回的最大接触数指定为一个大的数字来获取接触信息。
  collision_request.contacts = true;
  collision_request.max_contacts = 1000;
  //
  collision_result.clear();
  planning_scene.checkSelfCollision(collision_request, collision_result);
  RCLCPP_INFO_STREAM(LOGGER, "Test 5: Current state is " << (collision_result.collision ? "in" : "not in")
                                                         << " self collision");
  collision_detection::CollisionResult::ContactMap::const_iterator it;
  for (it = collision_result.contacts.begin(); it != collision_result.contacts.end(); ++it)
  {
    RCLCPP_INFO(LOGGER, "Contact between: %s and %s", it->first.first.c_str(), it->first.second.c_str());
  }

  // Modifying the Allowed Collision Matrix
  // 修改允许碰撞矩阵
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // AllowedCollisionMatrix（ACM）提供了一种机制，告诉碰撞世界忽略某些对象之间的碰撞：机器人的两部分和世界中的对象。
  // 我们可以告诉碰撞检查器忽略上述报告的所有链接之间的碰撞，即尽管链接实际上在碰撞中，但碰撞检查器将忽略这些碰撞，并返回机器人此特定状态下的不在碰撞中。
  // 注意，在这个例子中，我们还如何复制允许碰撞矩阵和当前状态，并将它们传递给碰撞检查函数。

  collision_detection::AllowedCollisionMatrix acm = planning_scene.getAllowedCollisionMatrix();
  moveit::core::RobotState copied_state = planning_scene.getCurrentState();

  collision_detection::CollisionResult::ContactMap::const_iterator it2;
  for (it2 = collision_result.contacts.begin(); it2 != collision_result.contacts.end(); ++it2)
  {
    acm.setEntry(it2->first.first, it2->first.second, true);
  }
  collision_result.clear();
  planning_scene.checkSelfCollision(collision_request, collision_result, copied_state, acm);
  RCLCPP_INFO_STREAM(LOGGER, "Test 6: Current state is " << (collision_result.collision ? "in" : "not in")
                                                         << " self collision");

  // Full Collision Checking
  // 完整碰撞检查
  // ~~~~~~~~~~~~~~~~~~~~~~~
  //
  // 虽然我们一直在检查自碰撞，但我们也可以使用checkCollision函数，这些函数将检查自碰撞和与环境（目前为空）的碰撞。这是您在规划器中最常使用的碰撞检查函数集。
  // 请注意，与环境的碰撞检查将使用机器人的填充版本。填充有助于将机器人与环境中的障碍物保持更远。
  collision_result.clear();
  planning_scene.checkCollision(collision_request, collision_result, copied_state, acm);
  RCLCPP_INFO_STREAM(LOGGER, "Test 7: Current state is " << (collision_result.collision ? "in" : "not in")
                                                         << " self collision");

  // Constraint Checking
  // 约束检查
  // ^^^^^^^^^^^^^^^^^^^
  //
  // PlanningScene类还包括用于检查约束的易于使用的函数调用。约束可以是两种类型之一：
  //（a）从KinematicConstraint集合中选择的约束：即JointConstraint、PositionConstraint、OrientationConstraint和VisibilityConstraint
  //（b）通过回调指定的用户定义约束。我们首先来看一个简单的KinematicConstraint的示例。
  //
  // Checking Kinematic Constraints
  // 检查运动学约束
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // 首先，我们将为Panda机器人的panda_arm组定义一个末端执行器的简单位置和方向约束。
  // 注意使用了方便的函数来填充约束（这些函数可以在moveit_core目录中的kinematic_constraints目录中的utils.h文件中找到）。

  std::string end_effector_name = joint_model_group->getLinkModelNames().back();

  geometry_msgs::msg::PoseStamped desired_pose;
  desired_pose.pose.orientation.w = 1.0;
  desired_pose.pose.position.x = 0.3;
  desired_pose.pose.position.y = -0.185;
  desired_pose.pose.position.z = 0.5;
  desired_pose.header.frame_id = "panda_link0";
  moveit_msgs::msg::Constraints goal_constraint =
      kinematic_constraints::constructGoalConstraints(end_effector_name, desired_pose);

  // 现在，我们可以使用PlanningScene类中的isStateConstrained函数来检查状态是否符合此约束。

  copied_state.setToRandomPositions();
  copied_state.update();
  bool constrained = planning_scene.isStateConstrained(copied_state, goal_constraint);
  RCLCPP_INFO_STREAM(LOGGER, "Test 8: Random state is " << (constrained ? "constrained" : "not constrained"));

  // 当您想一次又一次地检查相同的约束（例如在规划器中）时，有更有效的方法。我们首先构造一个KinematicConstraintSet，它预处理ROS约束消息并设置它以进行快速处理。

  kinematic_constraints::KinematicConstraintSet kinematic_constraint_set(kinematic_model);
  kinematic_constraint_set.add(goal_constraint, planning_scene.getTransforms());
  bool constrained_2 = planning_scene.isStateConstrained(copied_state, kinematic_constraint_set);
  RCLCPP_INFO_STREAM(LOGGER, "Test 9: Random state is " << (constrained_2 ? "constrained" : "not constrained"));

  // 有一种直接的方法可以使用 KinematicConstraintSet 类来实现这一点。

  kinematic_constraints::ConstraintEvaluationResult constraint_eval_result =
      kinematic_constraint_set.decide(copied_state);
  RCLCPP_INFO_STREAM(LOGGER, "Test 10: Random state is "
                                 << (constraint_eval_result.satisfied ? "constrained" : "not constrained"));

  // User-defined constraints
  // 用户定义的约束
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  // 现在，每当调用isStateFeasible时，都会调用此用户定义的回调。

  planning_scene.setStateFeasibilityPredicate(stateFeasibilityTestExample);
  bool state_feasible = planning_scene.isStateFeasible(copied_state);
  RCLCPP_INFO_STREAM(LOGGER, "Test 11: Random state is " << (state_feasible ? "feasible" : "not feasible"));

  // 每当调用isStateValid时，都会进行三个检查：(a)碰撞检查 (b)约束检查和 (c)使用用户定义的回调进行可行性检查。

  bool state_valid = planning_scene.isStateValid(copied_state, kinematic_constraint_set, "panda_arm");
  RCLCPP_INFO_STREAM(LOGGER, "Test 12: Random state is " << (state_valid ? "valid" : "not valid"));

  // 请注意，通过MoveIt和OMPL可用的所有规划器当前都会使用用户定义的回调执行碰撞检查、约束检查和可行性检查。

  rclcpp::shutdown();
  return 0;
}
```

## 输出

```text
blues@pasture-10:~/vscode_projects/cyan_demos/moveit_ws$ ros2 launch moveit2_tutorials planning_scene_tutorial.launch.py
[INFO] [launch]: All log files can be found below /home/blues/.ros/log/2025-01-24-14-23-35-317348-pasture-10-38088
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [planning_scene_tutorial-1]: process started with pid [38089]
[planning_scene_tutorial-1] [INFO] [1737699815.402008812] [moveit_rdf_loader.rdf_loader]: Loaded robot model in 0.00102517 seconds
[planning_scene_tutorial-1] [INFO] [1737699815.402037223] [moveit_robot_model.robot_model]: Loading robot model 'panda'...
[planning_scene_tutorial-1] [INFO] [1737699815.410305944] [planning_scene_tutorial]: Test 1: Current state is in self collision
[planning_scene_tutorial-1] [INFO] [1737699815.410370753] [planning_scene_tutorial]: Test 2: Current state is not in self collision
[planning_scene_tutorial-1] [INFO] [1737699815.410379780] [planning_scene_tutorial]: Test 3: Current state is not in self collision
[planning_scene_tutorial-1] [INFO] [1737699815.410382780] [planning_scene_tutorial]: Test 4: Current state is valid
[planning_scene_tutorial-1] [INFO] [1737699815.410423584] [planning_scene_tutorial]: Test 5: Current state is in self collision
[planning_scene_tutorial-1] [INFO] [1737699815.410425702] [planning_scene_tutorial]: Contact between: panda_leftfinger and panda_link1
[planning_scene_tutorial-1] [INFO] [1737699815.410427868] [planning_scene_tutorial]: Contact between: panda_link1 and panda_rightfinger
[planning_scene_tutorial-1] [INFO] [1737699815.410443083] [planning_scene_tutorial]: Test 6: Current state is not in self collision
[planning_scene_tutorial-1] [INFO] [1737699815.410452560] [planning_scene_tutorial]: Test 7: Current state is not in self collision
[planning_scene_tutorial-1] [INFO] [1737699815.410500323] [planning_scene_tutorial]: Test 8: Random state is not constrained
[planning_scene_tutorial-1] [INFO] [1737699815.410506990] [planning_scene_tutorial]: Test 9: Random state is not constrained
[planning_scene_tutorial-1] [INFO] [1737699815.410509393] [planning_scene_tutorial]: Test 10: Random state is not constrained
[planning_scene_tutorial-1] [INFO] [1737699815.410511847] [planning_scene_tutorial]: Test 11: Random state is not feasible
[planning_scene_tutorial-1] [INFO] [1737699815.410532895] [planning_scene_tutorial]: Test 12: Random state is not valid
[INFO] [planning_scene_tutorial-1]: process has finished cleanly [pid 38089]
```

## 启动文件

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("moveit_resources_panda").to_moveit_configs()

    # Planning Scene Tutorial executable
    planning_scene_tutorial = Node(
        name="planning_scene_tutorial",
        package="moveit2_tutorials",
        executable="planning_scene_tutorial",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
    )

    return LaunchDescription([planning_scene_tutorial])
```
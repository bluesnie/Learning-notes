###### datetime:2025/01/23 16:15

###### author:nzb

# [机器人模型和机器人状态](https://moveit.picknik.ai/main/doc/examples/robot_model_and_robot_state/robot_model_and_robot_state_tutorial.html#robot-model-and-robot-state)


机器人模型（[RobotModel](https://github.com/moveit/moveit2/blob/main/moveit_core/robot_model/include/moveit/robot_model/robot_model.hpp)）类和机器人状态（[RobotState](https://github.com/moveit/moveit2/blob/main/moveit_core/robot_state/include/moveit/robot_state/robot_state.hpp)）类是可以访问机器人运动学的核心类。 

- 机器人模型（RobotModel）类包含所有链接和关节之间的关系，包括从 URDF 加载的关节限位属性。 RobotModel 还将机器人的链接和关节分成 SRDF 中定义的规划组。 有关 URDF 和 SRDF 的单独教程，请点击此处： [URDF和SRDF教程](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html)
- 机器人状态（RobotState）包含机器人在某个时间点的信息，存储关节位置矢量，以及可选的速度和加速度矢量。 这些信息可用于获取依赖于机器人当前状态的运动学信息，例如末端效应器的雅各布因子。 RobotState 还包含一些辅助函数，用于根据末端效应器的位置（笛卡尔姿态）设置手臂位置和计算笛卡尔轨迹。

> 实例：`ros2 launch moveit2_tutorials robot_model_and_robot_state_tutorial.launch.py`

## 代码解读

```c++
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  // This enables loading undeclared parameters
  // best practice would be to declare parameters in the corresponding classes
  // and provide descriptions about expected use
  node_options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("robot_model_and_state_tutorial", node_options);
  const auto& LOGGER = node->get_logger();

  // BEGIN_TUTORIAL
  // Start
  // ^^^^^
  // Setting up to start using the RobotModel class is very easy. In
  // general, you will find that most higher-level components will
  // return a shared pointer to the RobotModel. You should always use
  // that when possible. In this example, we will start with such a
  // shared pointer and discuss only the basic API. You can have a
  // look at the actual code API for these classes to get more
  // information about how to use more features provided by these
  // classes.
  //
  // We will start by instantiating a
  // `RobotModelLoader`_
  // object, which will look up
  // the robot description on the ROS parameter server and construct a
  // :moveit_codedir:`RobotModel<moveit_core/robot_model/include/moveit/robot_model/robot_model.hpp>` for us to use.
  //
  // .. _RobotModelLoader:
  //     https://github.com/moveit/moveit2/blob/main/moveit_ros/planning/robot_model_loader/include/moveit/robot_model_loader/robot_model_loader.hpp
  robot_model_loader::RobotModelLoader robot_model_loader(node);
  const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader.getModel();
  RCLCPP_INFO(LOGGER, "Model frame: %s", kinematic_model->getModelFrame().c_str());

  // Using the :moveit_codedir:`RobotModel<moveit_core/robot_model/include/moveit/robot_model/robot_model.hpp>`, 
  // we can construct a :moveit_codedir:`RobotState<moveit_core/robot_state/include/moveit/robot_state/robot_state.hpp>` that maintains the configuration of the robot. We will set all joints in the state to their default values. We can then get a :moveit_codedir:`JointModelGroup<moveit_core/robot_model/include/moveit/robot_model/joint_model_group.hpp>`, which represents the robot model for a particular group, e.g. the "panda_arm" of the Panda robot.
  // 使用机器人模型，我们可以构建一个机器人状态（RobotState）来维护机器人的配置。 我们将把状态中的所有关节设置为默认值。 然后，我们可以得到一个关节模型组（JointModelGroup），它代表某个特定组的机器人模型，例如熊猫机器人的 "熊猫臂"（panda_arm）。
  moveit::core::RobotStatePtr robot_state(new moveit::core::RobotState(kinematic_model));
  robot_state->setToDefaultValues();
  const moveit::core::JointModelGroup* joint_model_group = kinematic_model->getJointModelGroup("panda_arm");

  const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();

  // Get Joint Values
  // ^^^^^^^^^^^^^^^^
  // We can retrieve the current set of joint values stored in the state for the Panda arm.
  std::vector<double> joint_values;
  robot_state->copyJointGroupPositions(joint_model_group, joint_values);
  for (std::size_t i = 0; i < joint_names.size(); ++i)
  {
    RCLCPP_INFO(LOGGER, "Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
  }

  // Joint Limits
  // ^^^^^^^^^^^^
  // setJointGroupPositions() 本身不会强制执行关节限制，但调用 enforceBounds() 可以实现。
  /* Set one joint in the Panda arm outside its joint limit */
  joint_values[0] = 5.57;
  robot_state->setJointGroupPositions(joint_model_group, joint_values);

  /* Check whether any joint is outside its joint limits */
  RCLCPP_INFO_STREAM(LOGGER, "Current state is " << (robot_state->satisfiesBounds() ? "valid" : "not valid"));

  /* Enforce the joint limits for this state and check again*/
  robot_state->enforceBounds();
  RCLCPP_INFO_STREAM(LOGGER, "Current state is " << (robot_state->satisfiesBounds() ? "valid" : "not valid"));

  // Forward Kinematics
  // ^^^^^^^^^^^^^^^^^^
  // Now, we can compute forward kinematics for a set of random joint
  // values. Note that we would like to find the pose of the
  // "panda_link8" which is the most distal link in the
  // "panda_arm" group of the robot.
  robot_state->setToRandomPositions(joint_model_group);
  const Eigen::Isometry3d& end_effector_state = robot_state->getGlobalLinkTransform("panda_link8");

  /* Print end-effector pose. Remember that this is in the model frame */
  RCLCPP_INFO_STREAM(LOGGER, "Translation: \n" << end_effector_state.translation() << "\n");
  RCLCPP_INFO_STREAM(LOGGER, "Rotation: \n" << end_effector_state.rotation() << "\n");

  // Inverse Kinematics
  // ^^^^^^^^^^^^^^^^^^
  // We can now solve inverse kinematics (IK) for the Panda robot.
  // To solve IK, we will need the following:
  //
  //  * The desired pose of the end-effector (by default, this is the last link in the "panda_arm" chain):
  //    end_effector_state that we computed in the step above.
  //  * The timeout: 0.1 s
  double timeout = 0.1;
  bool found_ik = robot_state->setFromIK(joint_model_group, end_effector_state, timeout);

  // Now, we can print out the IK solution (if found):
  if (found_ik)
  {
    robot_state->copyJointGroupPositions(joint_model_group, joint_values);
    for (std::size_t i = 0; i < joint_names.size(); ++i)
    {
      RCLCPP_INFO(LOGGER, "Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
    }
  }
  else
  {
    RCLCPP_INFO(LOGGER, "Did not find IK solution");
  }

  // Get the Jacobian
  // ^^^^^^^^^^^^^^^^
  // We can also get the Jacobian from the
  // :moveit_codedir:`RobotState<moveit_core/robot_state/include/moveit/robot_state/robot_state.hpp>`.
  Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
  Eigen::MatrixXd jacobian;
  robot_state->getJacobian(joint_model_group, robot_state->getLinkModel(joint_model_group->getLinkModelNames().back()),
                           reference_point_position, jacobian);
  RCLCPP_INFO_STREAM(LOGGER, "Jacobian: \n" << jacobian << "\n");
  // END_TUTORIAL

  rclcpp::shutdown();
  return 0;
}
```

## 输出

```text
[INFO] [launch]: All log files can be found below /home/blues/.ros/log/2025-01-23-17-20-02-828828-pasture-10-43955
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [robot_model_and_robot_state_tutorial-1]: process started with pid [43956]
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.910483504] [moveit_rdf_loader.rdf_loader]: Loaded robot model in 0.000824903 seconds
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.910503727] [moveit_robot_model.robot_model]: Loading robot model 'panda'...
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.914986406] [robot_model_and_state_tutorial]: Model frame: world
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915039339] [robot_model_and_state_tutorial]: Joint panda_joint1: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915043593] [robot_model_and_state_tutorial]: Joint panda_joint2: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915045723] [robot_model_and_state_tutorial]: Joint panda_joint3: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915047573] [robot_model_and_state_tutorial]: Joint panda_joint4: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915049343] [robot_model_and_state_tutorial]: Joint panda_joint5: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915051178] [robot_model_and_state_tutorial]: Joint panda_joint6: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915053132] [robot_model_and_state_tutorial]: Joint panda_joint7: 0.000000
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915055745] [robot_model_and_state_tutorial]: Current state is not valid
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915058370] [robot_model_and_state_tutorial]: Current state is valid
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915082856] [robot_model_and_state_tutorial]: Translation: 
[robot_model_and_robot_state_tutorial-1]  0.416676
[robot_model_and_robot_state_tutorial-1]  0.501069
[robot_model_and_robot_state_tutorial-1] -0.138612
[robot_model_and_robot_state_tutorial-1] 
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915090614] [robot_model_and_state_tutorial]: Rotation: 
[robot_model_and_robot_state_tutorial-1]  -0.674875   0.296534  -0.675731
[robot_model_and_robot_state_tutorial-1]   0.437197   0.898365 -0.0424092
[robot_model_and_robot_state_tutorial-1]   0.594477  -0.324048  -0.735928
[robot_model_and_robot_state_tutorial-1] 
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915132218] [robot_model_and_state_tutorial]: Joint panda_joint1: 0.727893
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915134928] [robot_model_and_state_tutorial]: Joint panda_joint2: 1.756749
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915136823] [robot_model_and_state_tutorial]: Joint panda_joint3: 0.214201
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915138709] [robot_model_and_state_tutorial]: Joint panda_joint4: -0.585159
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915140541] [robot_model_and_state_tutorial]: Joint panda_joint5: 0.239392
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915142356] [robot_model_and_state_tutorial]: Joint panda_joint6: 1.741185
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915144255] [robot_model_and_state_tutorial]: Joint panda_joint7: -2.066926
[robot_model_and_robot_state_tutorial-1] [INFO] [1737624002.915166286] [robot_model_and_state_tutorial]: Jacobian: 
[robot_model_and_robot_state_tutorial-1]   -0.501069   -0.352095   -0.215715    0.313214  -0.0409505    0.121742 -2.1684e-17
[robot_model_and_robot_state_tutorial-1]    0.416676   -0.313763    0.268989    0.163938   0.0735211    0.065999 1.38778e-17
[robot_model_and_robot_state_tutorial-1]           0   -0.644442   0.0952036    0.337704   0.0333641  0.00398945  1.9082e-17
[robot_model_and_robot_state_tutorial-1]           0   -0.665298    0.733707    0.620754     0.45903    0.452348   -0.675731
[robot_model_and_robot_state_tutorial-1]           0    0.746578    0.653829   -0.755662    0.566316   -0.812129  -0.0424092
[robot_model_and_robot_state_tutorial-1]           1 4.89664e-12   -0.184883   -0.208902   -0.684527   -0.368547   -0.735928
[robot_model_and_robot_state_tutorial-1] 
[INFO] [robot_model_and_robot_state_tutorial-1]: process has finished cleanly [pid 43956]
```

## 启动文件

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("moveit_resources_panda").to_moveit_configs()

    tutorial_node = Node(
        package="moveit2_tutorials",
        executable="robot_model_and_robot_state_tutorial",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
    )

    return LaunchDescription([tutorial_node])
```
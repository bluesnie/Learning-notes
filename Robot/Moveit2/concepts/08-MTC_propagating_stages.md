###### datetime:2025/04/27 16:39

###### author:nzb

# 传播阶段

传播器从一个相邻状态接收解决方案，解决问题，然后将结果传播给对面的相邻状态。
根据实现的不同，该阶段可以向前、向后或双向传递解决方案。
`MTC` 提供了以下传播阶段：
- ModifyPlanningScene
- MoveRelative
- MoveTo
- FixCollisionObjects

## ModifyPlanningScene

`ModifyPlanningScene` 阶段在不移动机器人的情况下对规划场景进行修改。

默认情况下，该阶段会向两个方向传播结果。

默认成本项为常数 0。

该阶段包含以下功能 
* 启用或禁用链接间的碰撞检查 
* 将对象附加或分离到机器人链接
* 从场景中生成或移除对象

附加对象的示例代码

```cpp
auto stage = std::make_unique<stages::ModifyPlanningScene>("attach object");
stage->attachObject("object_name", "gripper_frame_name");
```

启用碰撞的示例代码

```cpp
auto stage = std::make_unique<stages::ModifyPlanningScene>("Allow collision between object and gripper");
stage->allowCollisions("object_name", "gripper_frame_name", true);
```

[API doc for ModifyPlanningScene.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1ModifyPlanningScene.html)

## MoveRelative

`MoveRelative` 阶段用于执行笛卡尔运动。

默认情况下，该阶段向两个方向传播结果。

该阶段的默认规划时间为 1.0 秒。

默认成本项取决于路径长度。

由用户设置的属性表

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| group | void setGroup(std::string group) | Name of planning group. |
| ik_frame | void setIKFrame(std::string group) | Frame to be moved in Cartesian direction. |
| min_distance | void setMinDistance(double distance) | Minimum distance to move. Default is -1.0. |
| max_distance | void setMaxDistance(double distance) | Maximum distance to move. Default is 0.0. |
| path_constaints | void setPathConstraints(moveit_msgs/Constraints path_constaints) | Constraints to maintain during trajectory |
| direction | void setDirection(geometry_msgs/TwistStamped twist) | Perform twist motion on specified link. |
| direction | void setDirection(geometry_msgs/Vector3Stamped direction) | Translate link along given direction. |
| direction | void setDirection(std::map<std::string, double> direction) | Move specified joint variables by given amount |

[API doc for MoveRelative.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1MoveRelative.html)

示例代码

```cpp
const auto cartesian_planner = std::make_shared<moveit::task_constructor::solvers::CartesianPath>();
auto approach_pose =
    std::make_unique<moveit::task_constructor::stages::MoveRelative>("Approach", cartesian_planner);
// Propagate the solution backward only
stage_approach_grasp->restrictDirection(moveit::task_constructor::stages::MoveRelative::BACKWARD);
stage_approach_grasp->setGroup("manipulator");
stage_approach_grasp->setIKFrame("tool_frame");

// Move the end effector by 0.15m in the z direction.
const Eigen::Vector3d approach{ 0.0, 0.0, 0.15 };
geometry_msgs::msg::Vector3Stamped approach_vector;
tf2::toMsg(approach, approach_vector.vector);
approach_vector.header.frame_id = "tool_frame";
stage_approach_grasp->setDirection(approach_vector);
```

## MoveTo

`MoveTo` 阶段用于移动到联合状态或目标姿势。

默认情况下，该阶段会向两个方向传播结果。

该阶段的默认规划时间为 1.0 秒。

默认成本项取决于路径长度。

下表列出了该阶段需要设置的属性。
目标可以以不同格式指定。


| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| group | void setGroup(std::string group) | Name of planning group. |
| ik_frame | void setIKFrame(geometry_msgs/PoseStamped pose) | Frame to be moved towards goal pose. |
| goal | void setGoal(geometry_msgs/PoseStamped pose) | Move link to given pose |
| goal | void setGoal(geometry_msgs/PointStamped point) | Move link to given point, keeping current orientation |
| goal | void setGoal(std::string named_joint_pose) | Move joint model group to given named pose. The named pose should be described in the SRDF file. |
| goal | void setGoal(moveit_msgs/RobotState robot_state) | Move joints specified in msg to their target values. |
| goal | void setGoal(std::map<std::string, double> joints) | Move joints by name to their mapped target values. |
| path_constaints | void setPathConstraints(moveit_msgs:::Constraints path_constaints) | Constraints to maintain during trajectory |

[API doc for MoveTo.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1MoveTo.html)

示例代码

```cpp
const auto joint_interpolation_planner =
    std::make_shared<moveit::task_constructor::solvers::JointInterpolationPlanner>();
auto stage =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("close gripper", joint_interpolation_planner);
// Set trajectory execution info. This will contain the list of controllers used to actuate gripper and arm.
// Since this property is set during task initialization, we can inherit from it.
stage->properties().set("trajectory_execution_info",
                        std::any_cast<moveit::task_constructor::TrajectoryExecutionInfo>(task->properties().get("trajectory_execution_info")));
stage->setGroup("gripper");
stage->setGoal("closed"); // Group state named in SRDF
stage->setTimeout(2.0);
```

## FixCollisionObjects

`FixCollisionObjects` 阶段检查碰撞并在适用时解决碰撞。

默认情况下，该阶段会双向传播结果。

默认代价项为常数 0。

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| direction | void setDirection(geometry_msgs/Vector3 direction) | Direction vector to fix collision by shifting object along correction direction. A default direction is calculated if not explicitly set.(通过沿修正方向移动物体来修复碰撞的方向向量。如果没有明确设置，则会计算出默认方向。) |
| penetration | void setMaxPenetration(double penetration) | Cutoff length up to which collision objects get fixed. If the object’s collision length is greater than the value set, the collision will not be fixed.(固定碰撞对象的截止长度。如果对象的碰撞长度大于设定值，碰撞将不会被固定。) |

示例代码

```cpp
auto stage = std::make_unique<stages::FixCollisionObjects>();
stage->setMaxPenetration(0.04);
```

###### datetime:2025/04/27 16:39

###### author:nzb

# 生成阶段

生成阶段不从相邻阶段获得输入。它们计算结果并传递给相邻阶段。

`MTC` 提供以下生成器级：
- `CurrentState`
- `FixedState`
- `Monitoring Generators` - `GeneratePose`、`GenerateGraspPose`、`GeneratePlacePose` 和 `GenerateRandomPose`。

## CurrentState

`CurrentState` 阶段通过 `get_planning_scene` 服务获取当前的规划场景。
该阶段通常用于 `MTC` 任务流水线的开头，以便根据当前机器人状态设置起始状态。

示例代码

```cpp
auto current_state = std::make_unique<moveit::task_constructor::stages::CurrentState>("current_state");
```

[API doc for CurrentState.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1CurrentState.html)

## FixedState

固定状态阶段会生成一个预定义的规划场景状态。

```cpp
moveit::task_constructor::Task t;
auto node = rclcpp::Node::make_shared("node_name");
t.loadRobotModel(node);

auto scene = std::make_shared<planning_scene::PlanningScene>(t.getRobotModel());
auto& state = scene->getCurrentStateNonConst();
state.setToDefaultValues();  // initialize state
state.setToDefaultValues(state.getJointModelGroup("left_arm"), "home");
state.setToDefaultValues(state.getJointModelGroup("right_arm"), "home");
state.update();
spawnObject(scene); // User defined function to add a CollisionObject to planning scene

auto initial = std::make_unique<stages::FixedState>();
initial->setState(scene);
```

[API doc for FixedState.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1FixedState.html)

## Monitoring Generators

监控生成器有助于监控和使用另一个阶段的解决方案。

### GeneratePose

`GeneratePose` 是一个监控生成器阶段，可用于根据被监控阶段提供的解决方案生成姿势。

### GenerateGraspPose

`GenerateGraspPose` 阶段源于 `GeneratePose`，后者是一个监控生成器。
该阶段通常会监控 "当前状态 "阶段，因为该阶段需要最新的 "规划场景"（PlanningScene）来找到将围绕其生成抓取姿势的对象位置。
该阶段可通过设置所需的属性来生成抓取姿势。
设置同一属性可以有多种方法。例如，如下表所示，有两种功能可以设置预抓取姿势。用户可以通过使用字符串组状态或明确定义机器人状态来设置该属性。

由用户设置的属性表

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| eef | void setEndEffector(std::string eef) | Name of end effector |
| object | void setObject(std::string object) | Object to grasp. This object should exist in the planning scene. |
| angle_delta | void setAngleDelta(double delta) | Angular steps (rad). The target grasp pose is sampled around the object’s z axis |
| pregrasp | void setPreGraspPose(std::string pregrasp) | Pregrasp pose. For example, the gripper has to be in an open state before grasp. The pregrasp string here corresponds to the group state in SRDF. |
| pregrasp | void setPreGraspPose(moveit_msgs/RobotState pregrasp) | Pregrasp pose |
| grasp | void setGraspPose(std::string grasp) | Grasp pose |
| grasp | void setGraspPose(moveit_msgs/RobotState grasp) | Grasp pose |

有关代码的最新状态，请参阅 `API` 文档。`GenerateGraspPose` 的 [API 文档](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1GenerateGraspPose.html)。

示例代码

```cpp
auto initial_stage = std::make_unique<stages::CurrentState>("current state");
task->add(initial_stage);

auto gengrasp = std::make_unique<stages::GenerateGraspPose>("generate grasp pose");
gengrasp->setPreGraspPose("open");
gengrasp->setObject("object");
gengrasp->setAngleDelta(M_PI / 10.);
gengrasp->setMonitoredStage(initial_stage);
task->add(gengrasp);
```

### GeneratePlacePose

`GeneratePlacePose` 阶段源于 `GeneratePose`，后者是一个监控生成器。

该阶段为放置流水线生成姿势。

请注意，`GenerateGraspPose` 生成的姿势有一个角度`_delta` 间隔，而 `GeneratePlacePose` 的采样量是固定的，取决于对象的形状。

示例代码

```cpp
// Generate Place Pose
auto stage = std::make_unique<stages::GeneratePlacePose>("generate place pose");
stage->properties().configureInitFrom(Stage::PARENT, { "ik_frame" });
stage->properties().set("marker_ns", "place_pose");
stage->setObject(params.object_name);

// Set target pose
geometry_msgs::msg::PoseStamped p;
p.header.frame_id = params.object_reference_frame;
p.pose = vectorToPose(params.place_pose);
p.pose.position.z += 0.5 * params.object_dimensions[0] + params.place_surface_offset;
stage->setPose(p);
stage->setMonitoredStage(pick_stage_ptr);  // hook into successful pick solutions
```

[API doc for GeneratePlacePose.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1GeneratePlacePose.html)

### GenerateRandomPose

生成随机姿态（`GenerateRandomPose`）阶段源于生成姿态（`GeneratePose`），后者是一个监控生成器。
该阶段为姿态维度（`X/Y/Z/ROLL/PITCH/YAW`）配置随机数分布（见 https://en.cppreference.com/w/cpp/numeric/random）采样器，以随机化姿态。

由用户设置的属性表

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| max_solution | void setMaxSolution(size_t max_solution) | Limit of the number of spawned solutions in case randomized sampling is enabled. |

## FixedCartesianPose

固定笛卡尔姿态（`FixedCartesianPose`）生成一个固定的笛卡尔姿态。该阶段不进行采样。该阶段适用于从预期的未来状态进行规划，从而实现同步规划和执行等功能。










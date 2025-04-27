###### datetime:2025/04/27 16:39

###### author:nzb

# 连接阶段

`MTC` 只提供一个连接阶段，称为 "连接"。

连接阶段求解起始和目标状态之间的可行轨迹。

## Connect

连接阶段通过在相邻阶段给出的起点和终点目标之间寻找运动计划来连接两个阶段。

默认成本项取决于路径长度。

该阶段的默认规划时间为 1.0 秒。

由用户设置的属性表

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| merge_mode | void setGroup(std::string group) | 定义执行规划操作时使用的合并策略。该参数是一个枚举类型。可以是 SEQUENTIAL（存储连续轨迹）或 WAYPOINTS（通过航点连接轨迹）。默认为 WAYPOINTS。 |
| path_constaints | void setPathConstraints(moveit_msgs/Constraints path_constraints) | Constraints to maintain during trajectory |
| merge_time_parameterization |  | 默认为 TOTG（时间最优轨迹生成）。有关 TOTG 的信息，[请参阅时间参数化教程](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html#time-parameterization-algorithms) |

[API doc for Connect.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1Connect.html)

示例代码

```cpp
auto node = std::make_shared<rclcpp::Node>();
// planner used for connect
auto pipeline = std::make_shared<solvers::PipelinePlanner>(node, "ompl", "RRTConnectkConfigDefault");
// connect to pick
stages::Connect::GroupPlannerVector planners = { { "arm", pipeline }, { "gripper", pipeline } };
auto connect = std::make_unique<stages::Connect>("connect", planners);
```





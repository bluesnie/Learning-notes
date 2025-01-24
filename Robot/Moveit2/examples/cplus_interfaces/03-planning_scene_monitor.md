###### datetime:2025/01/24 14:15

###### author:nzb

# [规划场景监控器](https://moveit.picknik.ai/main/doc/examples/planning_scene_monitor/planning_scene_monitor_tutorial.html)

规划场景监控器（`PlanningSceneMonitor`）是维护最新规划场景的推荐接口。 机器人状态（`RobotState`）、当前状态监控器（`CurrentStateMonitor`）、规划场景（`PlanningScene`）、规划场景监控器（`PlanningSceneMonitor`）和规划场景接口（`PlanningSceneInterface`）之间的关系一开始可能会令人困惑。 本教程旨在阐明这些关键概念。

## RobotState

机器人状态(`RobotState`)是机器人的快照。 它包含机器人模型和一组关节值。

## CurrentStateMonitor

当前状态监控器（`CurrentStateMonitor`，简称 CSM）可视为机器人状态的 ROS 封装器。 它订阅所提供的 `JointState` 消息主题，该主题可提供单自由度执行器（如外旋或棱柱关节）的最新传感器值，并根据这些关节值更新其内部的 `RobotState`。 除了单自由度关节外，机器人还可能有浮动关节和平面关节等多自由度关节。 为了维护连接多自由度关节的链接和其他框架的最新变换信息，CSM 存储了一个 TF2 缓冲区，该缓冲区使用 TF2 变换监听器在其内部数据中设置变换。

## PlanningScene

规划场景（`PlanningScene`）是世界的一个快照，其中包括机器人状态（`RobotState`）和任意数量的碰撞对象。 规划场景可用于碰撞检查以及获取环境信息。

## PlanningSceneMonitor

规划场景监控器（`PlanningSceneMonitor`）用 ROS 接口封装了一个规划场景，用于保持规划场景的最新状态。 
要访问 `PlanningSceneMonitor` 的底层规划场景，请使用所提供的 `LockedPlanningSceneRW` 和 `LockedPlanningSceneRO` 类。 

`PlanningSceneMonitor` 有以下对象，它们都有自己的 ROS 接口，用于保持规划场景子组件的最新状态：

- 当前状态监控器(`CurrentStateMonitor`)用于通过机器人状态订阅器(`robot_state_subscriber_`)和缓冲区(`tf_buffer_`)跟踪机器人状态的更新，以及规划场景订阅器(`planning scene subscriber`)用于监听来自其他发布者的规划场景差异。 
- 占用图监控器(`OccupancyMapMonitor`)用于通过 ROS 主题和服务跟踪占用图的更新。

规划场景监控器(`PlanningSceneMonitor`)有以下订阅：

- `collision_object_subscriber_` - 侦听所提供主题的 CollisionObject 消息，这些消息可能会添加、删除或修改规划场景中的碰撞对象，并将这些对象传递到其监控的规划场景中 
- `planning_scene_world_subscriber_` - 侦听所提供主题的 PlanningSceneWorld 消息，这些消息可能包含碰撞对象信息和/或八维地图信息。 这对于保持规划场景监控器同步非常有用 
- `attached_collision_object_subscriber_` - 在提供的主题上监听 AttachedCollisionObject（附着碰撞对象）消息，该消息决定机器人状态中对象与链接的附着/脱落。

PlanningSceneMonitor 有以下服务： 
- `get_scene_service_` - 这是一个可选服务，用于获取完整的规划场景状态。

规划场景监视器的初始化条件是:

- `startSceneMonitor` - 启动 `planning_scene_subscriber_` ； 
- `startWorldGeometryMonitor` - 启动 `collision_object_subscriber_`、`planning_scene_world_subscriber_` 和 `OccupancyMapMonitor` ； 
- `startStateMonitor` - 启动 `CurrentStateMonitor` 和 `attached_collision_object_subscriber_`
- `startPublishingPlanningScene` - 启动另一个线程，在提供的主题上发布整个规划场景，供其他规划场景监控程序订阅
- `providePlanningSceneService` - 启动 `get_scene_service_`.

## PlanningSceneInterface

`PlanningSceneInterface` 是一个有用的类，可通过 `C++ API` 向 `MoveGroup` 的 `PlanningSceneMonitor` 发布更新，而无需创建自己的订阅者和服务客户端。 如果没有 `MoveGroup` 或 `MoveItCpp`，该类可能无法工作。
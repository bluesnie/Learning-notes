###### datetime:2025/02/08 14:37

###### author:nzb

# [可视化碰撞](https://moveit.picknik.ai/main/doc/examples/visualizing_collisions/visualizing_collisions_tutorial.html#visualizing-collisions)

本节将引导您通过 C++ 示例代码，允许您在 RViz 中移动和与机器人手臂交互时，可视化机器人自身与世界的碰撞接触点。

> 注意：该示例是ros1

## 运行代码

使用 `roslaunch` 启动文件直接从 `movelt_tutorials` 运行代码：

```bash
roslaunch moveri_tutorials visualizing_collisions_tutorial.launch
```

您现在应该会看到带有两个交互式标记的 Panda 机器人，您可以拖动这些标记。

## 类

本教程的代码主要在 `InteractiveRobot` 类中，我们将在下面逐步介绍。`InteractiveRobot` 类维护一个 `RobotModel`、一个 `RobotState` 以及关于“世界”的信息（在本例中，“世界”是一个黄色立方体）。

`InteractiveRobot` 类使用 `IMarker` 类来维护一个交互式标记。本教程不涵盖 `IMarker` 类（[`imarker.cpp`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/interactivity/src/imarker.cpp)）的实现，但大部分代码是从 [`basic_controls`](http://wiki.ros.org/rviz/Tutorials/Interactive%20Markers:%20Getting%20Started#basic_controls) 教程中复制的，如果您对交互式标记感兴趣，可以阅读更多相关内容。

## 交互

在 RViz 中，您将看到两组红/绿/蓝交互式标记箭头。用鼠标拖动这些箭头。移动右臂使其与左臂接触。您将看到标记接触点的洋红色球体。如果看不到洋红色球体，请确保您已按照上述说明添加了 `MarkerArray` 显示，并设置了 `interactive_robot_marray` 主题。此外，请确保将 `RobotAlpha` 设置为 0.3（或其他小于 1 的值），以便机器人透明且可以看到球体。移动右臂使其与黄色立方体接触（您也可以移动黄色立方体）。您将看到标记接触点的洋红色球体。

## 相关代码

完整的代码可以在 [`movelt_tutorials`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/visualizing_collisions) GitHub 项目中查看。使用的库可以在[这里](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/interactivity)找到。为了保持本教程的重点在碰撞接触上，省略了很多理解此演示所需的信息。要完全理解此演示，强烈建议您阅读源代码。

### 初始化规划场景和标记

在本教程中，我们使用 [`InteractiveRobot`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/interactivity/src/interactive_robot.cpp) 对象作为包装器，将 `robot_model` 与立方体和交互式标记结合在一起。我们还创建了一个用于碰撞检测的 `PlanningScene`。如果您尚未完成规划场景教程，请先完成该教程。

```cpp
InteractiveRobot robot;
g_planning_scene = new planning_scene::PlanningScene(robot.robotModel());
```

### 向规划场景添加几何体

```cpp
Eigen::Isometry3d world_cube_pose;
double world_cube_size;
robot.getWorldGeometry(world_cube_pose, world_cube_size);
g_word_cube.shape.reset(new shapes::Box(world_cube_size, world_cube_size, world_cube_size));
g_planning_scene->getWorldNonConst()->addToObject("world_cube", g_word_cube_shape, world_cube_pose);
```

### 碰撞请求

我们将为 Panda 机器人创建一个碰撞请求

```cpp
collision_detection::CollisionRequest c_req;
collision_detection::CollisionResult c_res;
c_req.group_name = robot.getGroupName();
c_req.contacts = true;
c_req.max_contacts = 100;
c_req.max_contacts_per_pair = 5;
c_req.verbose = false;
```

## 检查碰撞

我们检查机器人自身与世界的碰撞。

```cpp
g_planning_scene->checkCollision(c_req, c_res, *robot.robotState());
```

## 显示碰撞接触点

如果存在碰撞，我们获取接触点并将其显示为标记。

`getCollisionMarkersFromContacts()` 是一个辅助函数，它将碰撞接触点添加到 `MarkerArray` 消息中。如果您想将接触点用于显示以外的其他用途，可以遍历 `c_res.contacts`，这是一个接触点的 `std::map`。查看 [`collision_tools.cpp`](https://github.com/moveit/moveit/blob/noetic-devel/moveit_core/collision_detection/src/collision_tools.cpp) 中 `getCollisionMarkersFromContacts()` 的实现以了解如何操作。

```cpp
if (c_res.collision)
{
    ROS_INFO("COLLIDING contact_point_count=%d", (int)c_res.contact_count);
    if (c_res.contact_count > 0)
    {
        std_msgs::ColorRGBA color;
        color.r = 1.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 0.5;
        visualization_msgs::MarkerArray markers;

        /* 获取接触点并将其显示为标记 */
        collision_detection::getCollisionMarkersFromContacts(markers, "panda_link0",
        c_res.contacts, color,
        ros::Duration(), // 保留直到删除
        0.01);    // 半径
        publishMarkers(markers);
    }
}
```

# 启动文件

完整的启动文件可以在 [GitHub](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/visualizing_collisions) 上找到。本教程中的所有代码都可以从 `movelt_tutorials` 包中编译和运行。

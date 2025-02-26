###### datetime:2025/02/24 19:28

###### author:nzb

# [URDF 和 SRDF](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html#urdf-and-srdf)

## URDF

MoveIt 2 从 URDF（统一机器人描述格式）开始，这是 ROS 和 ROS 2 中描述机器人的原生格式。在本教程中，您将找到有关 URDF 的资源、重要提示以及 MoveIt 2 的特定要求。

### URDF 资源

- **[URDF ROS Wiki 页面](http://www.ros.org/wiki/urdf)**：这是关于 URDF 的主要信息来源。
- **[URDF 教程](https://docs.ros.org/en/rolling/Tutorials/URDF/URDF-Main.html)**：有关使用 URDF 的教程。
- **[SOLIDWORKS URDF 插件](http://www.ros.org/wiki/sw_urdf_exporter)**：允许您直接从 SOLIDWORKS 模型生成 URDF 的插件。

**注意**：尽管上述文档是为 ROS 编写的，但所有文档在将命令从 ROS 更改为 ROS 2（例如：`rosrun` -> `ros2 run` 或 `roslaunch` -> `ros2 launch`）后仍然有效。

### 重要提示

本节包含一组提示，确保您生成的 URDF 可以与 MoveIt 2 一起使用。在使用 MoveIt 2 之前，请确保您已阅读所有这些提示。

#### 关节名称中的特殊字符

关节名称不应包含以下特殊字符：`-`, `[`, `]`, `(`, `)`, `,`。

我们希望很快能够消除这些对关节名称的限制。

#### 安全限制

一些 URDF 除了关节限制外还设置了安全限制。以下是 Panda 头部旋转关节的安全控制器示例：

```xml
<safety_controller k_position="100" k_velocity="1.5" soft_lower_limit="-2.857" soft_upper_limit="2.857"/>
```

`soft_lower_limit` 和 `soft_upper_limit` 字段指定了该关节的位置限制。MoveIt 2 会将这些限制与 URDF 中指定的硬限制进行比较，并选择更保守的限制。

**注意**：如果 `soft_lower_limit` 和 `soft_upper_limit` 设置为 0.0，您的关节将无法移动。MoveIt 2 依赖于您指定正确的机器人模型。

#### 碰撞检测

MoveIt 2 使用 URDF 中指定的网格进行碰撞检测。URDF 允许您分别为可视化和碰撞检测指定两组网格。通常，可视化网格可以详细且美观，但碰撞网格应简单得多。网格中的三角形数量会影响碰撞检测所需的时间。整个机器人的三角形数量应在几千个左右。

#### 测试您的 URDF

测试您的 URDF 并确保一切正常非常重要。ROS URDF 包提供了一个 `check_urdf` 工具。要使用 `check_urdf` 工具验证您的 URDF，请按照[此处的说明操作](http://wiki.ros.org/urdf#Verification)。

### URDF 示例

有许多使用 ROS 的机器人 URDF 示例。

- **[URDF 示例](http://www.ros.org/wiki/urdf/Examples)**：来自 ROS 社区的 URDF 列表。

## SRDF

SRDF 或语义机器人描述格式是对 URDF 的补充，指定了关节组、默认机器人配置、额外的碰撞检测信息以及可能需要完全指定机器人姿态的额外变换。生成 SRDF 的推荐方法是使用 MoveIt 设置助手。

---

### 虚拟关节

URDF 仅包含有关机器人物理关节的信息。通常，需要定义额外的关节来指定机器人根链接相对于世界坐标系的姿态。在这种情况下，使用虚拟关节来指定此连接。例如，像 PR2 这样的移动机器人在平面上移动时，使用平面虚拟关节将世界坐标系附加到机器人的坐标系。固定机器人（如工业机械臂）应使用固定关节附加到世界。

### 被动关节

被动关节是机器人上的非驱动关节，例如差速驱动机器人上的被动脚轮。它们在 SRDF 中单独指定，以确保运动规划或控制管道中的不同组件知道这些关节无法直接控制。如果您的机器人有无动力的脚轮，应将它们指定为被动脚轮。

### 组

“组”（有时称为“关节组”或“规划组”）是 MoveIt 2 中的一个核心概念。MoveIt 2 始终作用于特定组。MoveIt 2 只会考虑移动规划组中的关节——其他关节保持静止。（可以通过创建一个包含所有关节的组来实现所有关节都可能移动的运动规划。）组只是关节和链接的集合。每个组可以通过以下几种方式指定：

#### 关节集合

组可以指定为关节的集合。每个关节的所有子链接都会自动包含在组中。

#### 链接集合

组也可以指定为链接的集合。所有链接的父关节也会包含在组中。

#### 串行链

串行链使用基链接和末端链接指定。链中的末端链接是链中最后一个关节的子链接。链中的基链接是链中第一个关节的父链接。

#### 子组集合

例如，您可以定义两个组 `left_arm` 和 `right_arm`，然后定义一个新组 `both_arms`，其中包含这两个组。

### 末端执行器

机器人中的某些组可以被指定为末端执行器。末端执行器通常通过固定关节连接到另一个组（如手臂）。请注意，在指定末端执行器组时，确保末端执行器与其连接的父组之间没有共同的链接。

### 自碰撞

默认自碰撞矩阵生成器（设置助手的一部分）搜索机器人上可以安全禁用碰撞检测的链接对，从而减少运动规划的处理时间。当这些链接对始终碰撞、从不碰撞、在机器人默认位置碰撞或链接在运动链上相邻时，它们会被禁用。采样密度指定了检查自碰撞的随机机器人位置的数量。较高的密度需要更多的计算时间，而较低的密度可能会禁用不应禁用的链接对。默认值为 10,000 次碰撞检查。碰撞检查是并行完成的，以减少处理时间。

### 机器人姿态

SRDF 还可以存储机器人的固定配置。一个典型的例子是定义机械臂的“HOME”位置。配置使用字符串 ID 存储，稍后可以使用该 ID 恢复配置。

### SRDF 文档

有关 SRDF 语法的更多信息，请阅读 [ROS SRDF Wiki](http://www.ros.org/wiki/srdf) 页面。

## 加载 URDF 和 SRDF

所有使用 RobotModel 的 MoveIt 组件都需要访问 URDF 和 SRDF 才能正常工作。在 ROS 1 中，这是通过将每个 XML 加载到全局参数服务器中的字符串参数（分别为 `/robot_description` 和 `/robot_description_semantic`）来实现的。ROS 2 没有全局参数服务器，因此确保所有相关节点都能访问这些参数需要更多的工作。

### 启动文件规范

一种选择是为每个需要参数的节点设置参数，通常如下所示：

加载 URDF 通常使用 `xacro`，因此加载方式如下：

```python
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command

robot_description = ParameterValue(Command(['xacro ', PATH_TO_URDF]), value_type=str)
```

同时，必须显式读取 SRDF。

```python
with open(PATH_TO_SRDF, 'r') as f:
    semantic_content = f.read()
```

然后必须将这些值加载到每个节点中。

```python
move_group_node = Node(
    package='moveit_ros_move_group',
    executable='move_group',
    output='screen',
    parameters=[{
        'robot_description': robot_description,
        'robot_description_semantic': semantic_content,
        # 更多参数
    }],
)
```

### 字符串主题规范

另一种方法是将这两个字符串作为主题发布。这种模式已经在 [`Robot State Publisher`](https://github.com/ros/robot_state_publisher/blob/37aff2034b58794b78f1682c8fab4d609f5d2e29/src/robot_state_publisher.cpp#L136) 中实现，它在 `/robot_description` 主题上发布 `std_msgs/msg/String` 消息。这可以在启动文件中完成：

```python
rsp_node = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    respawn=True,
    output='screen',
    parameters=[{
        'robot_description': robot_description,
        'publish_frequency': 15.0
    }]
)
```

您还可以告诉 MoveIt 节点发布该主题。

```python
output='screen',
parameters=[{
    'robot_description': robot_description,
    'publish_robot_description': True,
    # 更多参数
}],
```

发布机器人描述作为主题只需要完成一次，而不是在每个需要描述的节点中。

同样，我们也可以将 `SRDF` 作为 `std_msgs/msg/String` 消息发布。这需要在启动文件中设置一个节点的参数，并设置额外的参数 `publish_robot_description_semantic` 为 `True`。

```python
move_group_node = Node(
    package='moveit_ros_move_group',
    executable='move_group',
    output='screen',
    parameters=[{
        'robot_description_semantic': semantic_content,
        'publish_robot_description_semantic': True,
        # 更多参数
    }],
)
```

然后所有其他节点都可以订阅发布的消息。

### 底层：RDFLoader

在 MoveIt 代码的许多地方，使用 `RDFLoader` 类加载机器人描述和语义，该类将尝试从节点读取参数，如果失败，则尝试订阅字符串主题一段时间。如果两种方法都无法获取参数，则会在控制台上打印警告。

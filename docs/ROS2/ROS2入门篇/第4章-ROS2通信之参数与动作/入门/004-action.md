###### datetime:2023/09/18 10:18

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 4.动作（Action）通信与自定义接口

通过前面章节的学习，你已经掌握了ROS2中四大通信利器中话题、服务、参数这三个，还差最后一个就能将ROS2的通信机制全部打包带回家了，这节课就带你一起认识一下Action，并带你动手体验一下Action通信。

## 1.Action背景

前面章节学习了话题、服务、参数。

话题适用于节点间单向的频繁的数据传输，服务则适用于节点间双向的数据传递，而参数则用于动态调整节点的设置，动作Action和他们三个有什么不同之处呢？

如果这些问题体现在机器人上，可能是这样子的。我们通过服务服务发送一个目标点给机器人，让机器人移动到该点：

- 你不知道机器人有没有处理移动到目标点的请求（不能确认服务端接收并处理目标）
- 假设机器人收到了请求，你不知道机器人此时的位置和距离目标点的距离（没有反馈）
- 假设机器人移动一半，你想让机器人停下来，也没有办法通知机器人

上面的场景在机器人控制当中经常出现，比如控制导航程序，控制机械臂运动，控制小乌龟旋转等，很显然单个话题和服务不能满足我们的使用，因此ROS2针对控制这一场景，基于原有的话题和服务，设计了动作（Action）这一通信方式来解决这一问题。

## 2.Action的组成部分

知道了Action的出现原因，接着说说Action的三大组成部分目标、反馈和结果。

- 目标：即Action客户端告诉服务端要做什么，服务端针对该目标要有响应。解决了不能确认服务端接收并处理目标问题
- 反馈：即Action服务端告诉客户端此时做的进度如何（类似与工作汇报）。解决执行过程中没有反馈问题
- 结果：即Action服务端最终告诉客户端其执行结果，结果最后返回，用于表示任务最终执行情况。

> 参数是由服务构建出来了，而Action是由话题和服务共同构建出来的（一个Action = 三个服务+两个话题）
> - 三个服务分别是：
    >
- 1.目标传递服务
>   - 2.结果传递服务
>   - 3.取消执行服务
> - 两个话题：
    >
- 1.反馈话题（服务端发布，客户端订阅）
>   - 2.状态话题（服务端发布，客户端订阅）



![../_images/行动-单一行动.gif](imgs/Action-SingleActionClient.gif)

## 3.感受Action

带着前面对Action的了解，接着我们一起来了直观的通过小乌龟的案例来感受一下Action的魅力。

### 3.1 启动乌龟模拟器和键盘控制节点

乌龟模拟器

```
ros2 run turtlesim turtlesim_node
```

键盘控制节点

```
ros2 run turtlesim turtle_teleop_key
```

打开键盘控制节点后，你应该窗口中可以看到下面的提示

```
Use arrow keys to move the turtle.
Use G|B|V|C|D|E|R|T keys to rotate to absolute orientations. 'F' to cancel a rotation.
```

有请翻译官（其实用Deppl翻译的）

```
使用方向键移动乌龟。
用G、B、V、C、D、E、R、T键旋转到绝对方向。'F'可以取消旋转。
```

这段提示什么意思呢？其实就是字面的意思，

小乌龟键盘控制节点，提供两种可选的控制方式。

- 方向键，通过话题(Topic)控制小乌龟的（直接发送移动话题）
- 绝对旋转，则是采用动作(Action）来控制的小乌龟

### 3.2 使用绝对旋转(Action)控制小乌龟

使用绝对旋转控制小乌龟即使用Action来控制小乌龟。

在小乌龟的遥控窗口我们使用键盘上的F按键周围的按键来尝试运行控制下小乌龟的方向，你会看到小乌龟根据我们所按下按键所在的方向来在原地进行旋转。

同时在旋转的过程中，我们可以使用F按键，来取消小乌龟的运动。

## 4. Action的CLI工具

Action的命令行工具一共有三个,下面我们一一介绍。

### 4.1 action list

该命令用于获取目前系统中的action列表。

```
ros2 action list
```

你将看到

```
/turtle1/rotate_absolute
```

如果在list后加入-t参数，即可看到action的类型

```
/turtle1/rotate_absolute [turtlesim/action/RotateAbsolute]
```

知道了接口类型之后，可以使用接口相关CLI指令获取接口的信息

```
ros2 interface show turtlesim/action/RotateAbsolute 
```

结果

```
# The desired heading in radians
float32 theta
---
# The angular displacement in radians to the starting position
float32 delta
---
# The remaining rotation in radians
float32 remaining
```

### 4.2 action info

查看action信息，在终端中输入下面的指令。

```
ros2 action info /turtle1/rotate_absolute 
```

你将看到,action客户端和服务段的数量以及名字。

```
Action: /turtle1/rotate_absolute
Action clients: 1
    /teleop_turtle
Action servers: 1
    /turtlesim
```

### 4.3 action send_goal

该指令用于发送actin请求到服务端，我们可以模拟下，让小乌龟转到我们定义的角度。

我们只需要把goal发给服务端即可，根据goal的定义，我们可以看到goal是由一个浮点类型的theta组成的，我们把theta发给服务端。

发送弧度制1.6给小乌龟

```
ros2 action send_goal /turtle1/rotate_absolute turtlesim/action/RotateAbsolute "{theta: 1.6}"
```

结果

```
Waiting for an action server to become available...
Sending goal:
     theta: 1.6

Goal accepted with ID: b215ad060899444793229171e76481c7

Result:
    delta: -1.5840003490447998

Goal finished with status: SUCCEEDED
```

我们可以看到goal和result，但是没有看到feedback，这里我们需要加一个参数 --feedback

加上这个参数我们再发送一次。

```
ros2 action send_goal /turtle1/rotate_absolute turtlesim/action/RotateAbsolute "{theta: 1.5}" --feedback
```

可以看到了，这次的日志中多出了很多实时的反馈，反馈的数值是小乌龟当前的角度与我们给定的目标角度之间的差值，可以看到一直在变小。

```
Waiting for an action server to become available...
Sending goal:
     theta: 1.5

Feedback:
    remaining: -0.0840003490447998

Goal accepted with ID: b368de0ed1a54e00890f1b078f4671c8

Feedback:
    remaining: -0.06800031661987305

Feedback:
    remaining: -0.05200028419494629

Feedback:
    remaining: -0.03600025177001953

Feedback:
    remaining: -0.020000219345092773

Feedback:
    remaining: -0.004000186920166016

Result:
    delta: 0.08000016212463379

Goal finished with status: SUCCEEDED
```

5.自定义通信接口

我们接下来以控制机器人移动到点为例子，来学习Action通信。因为这个场景是我们自己定义的，ROS2并没有拿来就用的接口，所以我们需要自定义Action通信接口。

5.1 创建接口功能包

创建功能包

```shell
cd chapt4_ws/
ros2 pkg create robot_control_interfaces --build-type ament_cmake --destination-directory src --maintainer-name "fishros" --maintainer-email "fishros@foxmail.com"
```

创建接口文件

```shell
mkdir -p src/robot_control_interfaces/action
touch src/robot_control_interfaces/action/MoveRobot.action
```

packages.xml

```xml

<depend>rosidl_default_generators</depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

CMakeLists.txt

```cmake
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveRobot.action"
)
```

5.2 编写接口

```
# Goal: 要移动的距离
float32 distance
---
# Result: 最终的位置
float32 pose
---
# Feedback: 中间反馈的位置和状态
float32 pose
uint32 status
uint32 STATUS_MOVEING = 3
uint32 STATUS_STOP = 4
```

5.3 编译生成接口

```shell
colcon build --packages-select robot_control_interfaces
```

编译成功后，即可看到C++头文件和Python相关文件

- C++ Action消息头文件目录`install/robot_control_interfaces/include`
- Python Action消息文件目录`install/robot_control_interfaces/local/lib/python3.10`

6.总结

本节我们学习了Action并手动创建了Action的接口，下一节将带你一起使用接口完成任务。
###### datetime:2023/10/12 11:03

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# TODO

# 4.使用Nav2导航API进行导航

Nav2的API其实是Nav2提供的一个Python库，通过该库你可以事先调用你的机器人进行简单的控制（比如导航到点）。

很遗憾的是，该功能包官方并没有发布foxy版本的，再下一个galactic版本才开始正式发布。

从2022年5月23号开始，教程将开始向humble版本迁移，该部分内容将在humble版本发布。

## 1.导入nav2_simple_commander

```
from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from copy import deepcopy
```

初始化BasicNavigator

```
rclpy.init()
nav = BasicNavigator()
navigator.waitUntilNav2Active()
```

## 2.初始化位置

```
# ======================初始化位置，代替rviz2的2D Pose Estimate===============================
initial_pose = PoseStamped()
initial_pose.header.frame_id = 'map'
initial_pose.header.stamp = navigator.get_clock().now().to_msg()
initial_pose.pose.position.x = 0.0
initial_pose.pose.position.y = 0.0
initial_pose.pose.orientation.w = 1.0
navigator.setInitialPose(initial_pose)
```

## 3.导航到点

```
#========================导航到目标点1===========================================
goal_pose1 = deepcopy(initial_pose)
goal_pose1.pose.position.x = 1.5
nav.goToPose(goal_pose1)
while not nav.isNavComplete():
  feedback = nav.getFeedback()
  #检查是否超时，超时则停止导航到点   
  if feedback.navigation_duration > 600:
    nav.cancelNav()


#================================导航到目标点2==================================
goal_pose2 = deepcopy(initial_pose)
goal_pose2.pose.position.x = -1.5

nav.goToPose(goal_pose2)
while not nav.isNavComplete():
  feedback = nav.getFeedback()
  #检查是否超时，超时则停止导航到点   
  if feedback.navigation_duration > 600:
    nav.cancelNav()

#===============================查看返回结果=====================================
result = nav.getResult()
if result == NavigationResult.SUCCEEDED:
    print('Goal succeeded!')
elif result == NavigationResult.CANCELED:
    print('Goal was canceled!')
elif result == NavigationResult.FAILED:
    print('Goal failed!')
```

## 4.总结

上面是对nav2_simple_commander的简单介绍，介于本教程目前主要维护foxy版本问题，暂时该部分无法使用，不过你可以使用galactic版本进行实践。

--------------

###### datetime:2023/09/25 10:22

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)
> 
> [ros2 examples](https://github.com/ros2/examples)

# rclpy：节点基础知识

大多数节点都有发布者和订阅者，它们都是通过调用实例的函数创建的：

```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MyNode(Node):

    def __init__(self):
        super().__init__('my_node_name')

        self.publisher = self.create_publisher(String, 'output_topic', 10)
        self.subscription = self.create_subscription(
            String,
            'input_topic',
            self.callback,
            10)

    def callback(self, msg):
        self.get_logger().info("Recieved: %s" % msg.data)
        self.publisher.publish(msg)


if __name___ == "__main__":
    rclpy.init()
    my_node = MyNode()
    rclpy.spin(my_node)
    my_node.destroy_node()  # cleans up pub-subs, etc
    rclpy.shutdown()
```

## 关闭Handle

ROS1 有rospy.on_shutdown() - 但是 [不等于ROS2也有](https://github.com/ros2/rclpy/issues/244) ，不过，这真的不需要，因为我们手动关闭了一些东西，而不是像 rospy
那样使用了许多全局变量:

```python
try:
    rclpy.spin(my_node)
except KeyboardInterrupt:
    pass
finally:
    my_node.on_shutdown()  # do any custom cleanup
    my_node.destroy_node()
    rclpy.shutdown()
```

# rclpy：参数

```python
# node is rclpy.node.Node instance
# 42 is a great default for a parameter
node.declare_parameter("my_param_name", 42)

# To get the value:
param = node.get_parameter("my_param_name").value
```

## 动态参数

在 ROS2 中，所有参数都可以通过 ROS2 服务动态更新(不需要像动态重新配置那样定义重复内容)。

```python
from rcl_interfaces.msg import SetParametersResult

import rclpy
from rclpy.node import Node


class MyNode(Node):

    def __init__(self):
        super().__init__('my_node_name')

        # Declare a parameter
        self.declare_parameter("my_param_name", 42)

        # Then create callback
        self.set_parameters_callback(self.callback)

    def callback(self, parameters):
        result = SetParametersResult(successful=True)

        for p in parameters:
            if p.name == "my_param_name":
                if p.type_ != p.Type.INTEGER:
                    result.successful = False
                    result.reason = 'my_param_name must be an Integer'
                    return result
                if p.value < 20:
                    result.successful = False
                    result.reason = "my_param_name must be >= 20"
                    return result

        # Return success, so updates are seen via get_parameter()
        return result
```

有关调用 SET_PARAMETERS 服务的示例，请参阅 [ROS Answers](https://answers.ros.org/question/308541/ros2-rclpy-set-parameter-example/)

# RCLPY：TF2

TF2 库提供了对转换的轻松访问。以下所有示例都需要对 ros_Package 的依赖关系。

## 监听转换

```python
import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
```

## 应用变换

TF2 可以通过提供实现的包进行扩展。GEOMETRY_msgs 程序包为 msgs_ 提供这些。下面的示例使用 msgs.msg.PointStamed_，但这应该适用于 msgs_ 中的任何数据类型：

```python
from geometry_msgs.msg import PointStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

# Setup buffer/listener as above

p1 = PointStamped()
p1.header.frame_id = "source_frame"
# fill in p1

p2 = buffer.transform(p1, "target_frame")
```

## 变换

在 ROS1 中，Tf 包括模块。TF2 没有类似的模块。建议使用 Transforms3d Python 包，可通过 pip 获取：

```
sudo pip3 install transforms3d
```

例如，要旋转点：

```python
import numpy as np
from transforms3d.quaternion import quat2mat

# Create rotation matrix from quaternion
R = quat2mat([w, x, y, z])
# Create a vector to rotate
V = np.array([x, y, z]).reshape((3, 1))
# Rotate the vector
M = np.dot(R, V)

p = PointStamped()
p.point.x = M[0, 0]
p.point.y = M[1, 0]
p.point.z = M[2, 0]
```

# rclpy: Time

要获得相当于 rospy.Time.now()的内容，你现在需要一个 ROS2 节点：

```python
import rclpy
from rclpy.node import Node


class MyNode(Node):

    def some_func(self):
        t = self.get_clock().now()
        msg.header.stamp = t.to_msg()
```

从持续时间转换为消息很常见：

```python
import rclpy
from rclpy.duration import Duration

msg.duration = Duration(seconds=1).to_msg()
```

计时器是从节点创建的：

```python
import rclpy
from rclpy.node import Node


class MyNode(Node):

    def __init__(self):
        super().__init__("my_node")

        # Create a timer that fires every quarter second
        timer_period = 0.25
        self.timer = self.create_timer(timer_period, self.callback)

    def callback(self):
        self.get_logger().info("timer has fired")
```
###### datetime:2023/11/03 10:52

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 15.采用MicroROS发布里程计

获得了里程计数据后，下一步就是将里程计通过MicroROS话题发布到ROS 2 系统中。

## 一、了解接口

在 ROS 2 已有的消息接口中：

```
nav_msgs/msg/Odometry
```

用于表示里程计数据，该接口内容如下：

```
 ros2 interface show nav_msgs/msg/Odometry 
 ---
# This represents an estimate of a position and velocity in free space.
# The pose in this message should be specified in the coordinate frame given by header.frame_id
# The twist in this message should be specified in the coordinate frame given by the child_frame_id

# Includes the frame id of the pose parent.
std_msgs/Header header
        builtin_interfaces/Time stamp
                int32 sec
                uint32 nanosec
        string frame_id

# Frame id the pose points to. The twist is in this coordinate frame.
string child_frame_id

# Estimated pose that is typically relative to a fixed world frame.
geometry_msgs/PoseWithCovariance pose
        Pose pose
                Point position
                        float64 x
                        float64 y
                        float64 z
                Quaternion orientation
                        float64 x 0
                        float64 y 0
                        float64 z 0
                        float64 w 1
        float64[36] covariance

# Estimated linear and angular velocity relative to child_frame_id.
geometry_msgs/TwistWithCovariance twist
        Twist twist
                Vector3  linear
                        float64 x
                        float64 y
                        float64 z
                Vector3  angular
                        float64 x
                        float64 y
                        float64 z
        float64[36] covariance
```

注意看，除了表示位置的 pose 和表示速度的 twist ，还有 child_frame_id 这一参数，它表示里程计子坐标系名称，根据ROS 导航堆栈定义，一般用 base_link 或者 base_footprint 。

接着我们来编写代码。

## 二、编写代码

如何发布话题在前面的章节中我们已经学习过了，现在我们来编写代码，因为是直接在原来的代码基础上修改的，所以下面展示的代码前如果是+表示新增行，-表示删除行，没有符号表示没有修改。

### Kinematics.h

首先是/lib/Kinematics/Kinematics.h，增加四元数定义和欧拉角转四元数函数，这是因为ROS中姿态的表示使用的是四元数。

```cpp
+typedef struct
+{
+    float w;
+    float x;
+    float y;
+    float z;
+} quaternion_t;
+
 /**
  * @brief 里程计相关信息，根据轮子速度信息和运动模型推算而来
  *
@@ -41,6 +49,7 @@ typedef struct
     float x;                 // 坐标x
     float y;                 // 坐标y
     float yaw;               // yaw
+    quaternion_t quaternion; // 姿态四元数
     float linear_speed;      // 线速度
     float angular_speed;     // 角速度
 } odom_t;
@@ -56,6 +65,7 @@ private:
 public:
     Kinematics(/* args */) = default;
     ~Kinematics() = default;
+    static void Euler2Quaternion(float roll, float pitch, float yaw, quaternion_t &q);
 
     static void TransAngleInPI(float angle,float& out_angle);
```

### Kinematics.cpp

接着是：lib/Kinematics/Kinematics.cpp，增加 Euler2Quaternion 函数实现，在 odom 函数中增加对 Euler2Quaternion 函数的调用。

```cpp
 #include "Kinematics.h"
 
+// 用于将欧拉角转换为四元数。
+void Kinematics::Euler2Quaternion(float roll, float pitch, float yaw, quaternion_t &q)
+{
+    // 传入机器人的欧拉角 roll、pitch 和 yaw。
+    // 计算欧拉角的 sin 和 cos 值，分别保存在 cr、sr、cy、sy、cp、sp 六个变量中    
+    // https://blog.csdn.net/xiaoma_bk/article/details/79082629
+    double cr = cos(roll * 0.5);
+    double sr = sin(roll * 0.5);
+    double cy = cos(yaw * 0.5);
+    double sy = sin(yaw * 0.5);
+    double cp = cos(pitch * 0.5);
+    double sp = sin(pitch * 0.5);
+    // 计算出四元数的四个分量 q.w、q.x、q.y、q.z
+    q.w = cy * cp * cr + sy * sp * sr;
+    q.x = cy * cp * sr - sy * sp * cr;
+    q.y = sy * cp * sr + cy * sp * cr;
+    q.z = sy * cp * cr - cy * sp * sr;
+}


 odom_t &Kinematics::odom()
 {
+    // 调用 Euler2Quaternion 函数，将机器人的欧拉角 yaw 转换为四元数 quaternion。
+    Kinematics::Euler2Quaternion(0, 0, odom_.yaw, odom_.quaternion);
     return odom_;
 }
```

### main.cpp

接着修改了 /src/main.cpp ，主要添加了一个发布者，接着对时间进行同步，方便发布里程计话题时使用当前的时间。

然后对数据的各项进行设置，最后添加了里程计数据的发布，间隔 50ms 进行发布。

```cpp
 #include <Esp32McpwmMotor.h>         // 包含使用 ESP32 的 MCPWM 硬件模块控制 DC 电机的 ESP32 MCPWM 电机库
 #include <PidController.h>           // 包含 PID 控制器库，用于实现 PID 控制
 #include <Kinematics.h>              // 运动学相关实现
+#include <nav_msgs/msg/odometry.h>
+#include <micro_ros_utilities/string_utilities.h>
+rcl_publisher_t odom_publisher;   // 用于发布机器人的里程计信息（Odom）
+nav_msgs__msg__Odometry odom_msg; // 机器人的里程计信息
 
 Esp32PcntEncoder encoders[2];      // 创建一个长度为 2 的 ESP32 PCNT 编码器数组
 rclc_executor_t executor;          // 创建一个 RCLC 执行程序对象，用于处理订阅和发布
 
void microros_task(void *param)
 {
+  // 使用 micro_ros_string_utilities_set 函数设置到 odom_msg.header.frame_id 中
+  odom_msg.header.frame_id = micro_ros_string_utilities_set(odom_msg.header.frame_id, "odom");
+  odom_msg.child_frame_id = micro_ros_string_utilities_set(odom_msg.child_frame_id, "base_link");
   // 等待 2 秒，以便网络连接得到建立。
   delay(2000);

+  rclc_publisher_init_best_effort(
+      &odom_publisher,
+      &node,
+      ROSIDL_GET_MSG_TYPE_SUPPORT(nav_msgs, msg, Odometry),
+      "odom");
   // 设置 micro-ROS 执行器，并将订阅添加到其中。
   rclc_executor_init(&executor, &support.context, 1, &allocator);
   rclc_executor_add_subscription(&executor, &subscriber, &sub_msg, &twist_callback, ON_NEW_DATA);

      // 循环运行 micro-ROS 执行器以处理传入的消息。
      while (true)
      {
+       if (!rmw_uros_epoch_synchronized())
+       {
+         rmw_uros_sync_session(1000);
+         // 如果时间同步成功，则将当前时间设置为MicroROS代理的时间，并输出调试信息。
+         delay(10);
+       }
        delay(100);
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
      }
  }
 
 unsigned long previousMillis = 0; // 上一次打印的时间
-unsigned long interval = 1000;    // 间隔时间，单位为毫秒
+unsigned long interval = 50;      // 间隔时间，单位为毫秒
 
 void loop()
 {
     previousMillis = currentMillis; // 记录上一次打印的时间
     float linear_speed, angle_speed;
     kinematics.kinematic_forward(kinematics.motor_speed(0), kinematics.motor_speed(1), linear_speed, angle_speed);
-    Serial.printf("[%ld] linear:%f angle:%f\n", currentMillis, linear_speed, angle_speed);                       // 打印当前时间
-    Serial.printf("[%ld] x:%f y:%f yaml:%f\n", currentMillis,kinematics.odom().x, kinematics.odom().y, kinematics.odom().yaw); // 打印当前时间
+    // Serial.printf("[%ld] linear:%f angle:%f\n", currentMillis, linear_speed, angle_speed);                       // 打印当前时间
+    // Serial.printf("[%ld] x:%f y:%f yaml:%f\n", currentMillis,kinematics.odom().x, kinematics.odom().y, kinematics.odom().yaw); // 打印当前时间
+    // 用于获取当前的时间戳，并将其存储在消息的头部中
+    int64_t stamp = rmw_uros_epoch_millis();
+    // 获取机器人的位置和速度信息，并将其存储在一个ROS消息（odom_msg）中
+    odom_t odom = kinematics.odom();
+    odom_msg.header.stamp.sec = static_cast<int32_t>(stamp / 1000); // 秒部分
+    odom_msg.header.stamp.nanosec = static_cast<uint32_t>((stamp % 1000) * 1e6); // 纳秒部分
+    odom_msg.pose.pose.position.x = odom.x;
+    odom_msg.pose.pose.position.y = odom.y;
+    odom_msg.pose.pose.orientation.w = odom.quaternion.w;
+    odom_msg.pose.pose.orientation.x = odom.quaternion.x;
+    odom_msg.pose.pose.orientation.y = odom.quaternion.y;
+    odom_msg.pose.pose.orientation.z = odom.quaternion.z;
+
+    odom_msg.twist.twist.angular.z = odom.angular_speed;
+    odom_msg.twist.twist.linear.x = odom.linear_speed;
+
+    rcl_publish(&odom_publisher, &odom_msg, NULL);
   }
 

```

这三个文件修改好，接着就可以下载代码进行测试了。

## 三、下载测试

下载代码，运行agent，点击RST按键。

```shell
sudo docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO udp4 --port 8888 -v6
```

![image-20230306023859873](imgs/image-20230306023859873.png)

看到连接建立表示通信成功，接着用`ros2 topic list`

```shell
ros2 topic list
---
/cmd_vel
/odom
/parameter_events
/rosout
```

接着我们可以查看里程计数据或其发布频率。

```
ros2 topic echo /odom --once # 查看数据
---
header:
  stamp:
    sec: 2093
    nanosec: 40
  frame_id: odom
child_frame_id: base_link
pose:
  pose:
    position:
      x: 0.0
      y: 0.0
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
  covariance:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
twist:
  twist:
    linear:
      x: 0.0
      y: 0.0
      z: 0.0
    angular:
      x: 0.0
      y: 0.0
      z: 0.0
  covariance:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
---
```

查看数据频率

```
ros2 topic hz /odom
---
average rate: 19.376
        min: 0.047s max: 0.063s std dev: 0.00326s window: 21
average rate: 19.558
        min: 0.039s max: 0.063s std dev: 0.00338s window: 41
average rate: 19.527
        min: 0.039s max: 0.063s std dev: 0.00307s window: 61
average rate: 19.533
        min: 0.039s max: 0.063s std dev: 0.00301s window: 81
```

查看数据带宽

```
ros2 topic bw /odom
---
Subscribed to [/odom]
14.78 KB/s from 20 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
14.26 KB/s from 39 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
14.34 KB/s from 59 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
14.19 KB/s from 78 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
14.25 KB/s from 98 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
14.18 KB/s from 100 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
14.25 KB/s from 100 messages
	Message size mean: 0.72 KB min: 0.72 KB max: 0.72 KB
```

## 四、总结

有了控制话题和里程计话题，底盘部分就完成的差不多了，但是对于一个合格的底盘来说，其实还有很多待完善的地方，下一节我们就来说说可以怎么完善，以及完善的后的代码。
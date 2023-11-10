###### datetime:2023/09/25 10:22

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# Rclcpp：节点和组件

## 创建组件

```cpp
#include <rclcpp/rclcpp.hpp>

namespace my_pkg
{

class MyComponent : public rclcpp::Node
{
public:
  MyComponent(const rclcpp::NodeOptions& options)
  : rclcpp::Node("node_name", options)
  {
    // Note: you cannot use shared_from_this()
    //       here because the node is not fully
    //       initialized.
  }
};

}  // namespace my_pkg

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(my_pkg::MyComponent)
```

CMakeLists.txt 中的：

```cmake
add_library(my_component SHARED
  src/my_component.cpp
)
ament_target_dependencies(my_component
  rclcpp
  rclcpp_components
)

# Also add a node executable which simply loads the component
rclcpp_components_register_node(my_component
  PLUGIN "my_pkg::MyComponent"
  EXECUTABLE my_node
)
```

## Executors

要在线程中运行执行器，请执行以下操作：

```cpp
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>

rclcpp::executors::SingleThreadedExecutor executor;

// Node is rclcpp::Node::SharedPtr instance
executor.add_node(node);
std::thread executor_thread(
  std::bind(&rclcpp::executors::SingleThreadedExecutor::spin,
            &executor));
```

# Rclcpp：参数

需要声明参数。同时，如果你不打算稍后再次更新值，则可以获得该值：

```cpp
// node is an instance of rclcpp::Node
// 42 is a great default for a parameter
int param = node.declare_parameter<int>("my_param_name", 42);
```

要获取值，请执行以下操作：

```cpp
int param;
node.get_parameter("my_param_name", param);
```

## 动态参数

在 ROS2 中，所有参数都可以通过 ROS2 服务动态更新(不需要像动态重新配置那样定义重复内容)。

下面的例子适用于 eloquent 或更高版本(较早的 ROS2 版本只支持单个回调，并且有一个略有不同的 API)。有关有效类型的信息，请参阅的文档。

```cpp
#include <vector>
#include <rclcpp/rclcpp.hpp>

class MyNode : public rclcpp::Node
{
public:
  MyNode()
  {
    // Declare parameters first

    // Then create callback
    param_cb_ = this->add_on_set_parameters_callback(
      std::bind(&MyNode::updateCallback, this, std::placeholders::_1));
  }

private:
  // This will get called whenever a parameter gets updated
  rcl_interfaces::msg::SetParametersResult updateCallback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const rclcpp::Parameter & param : parameters)
    {
      if (param.get_name() == "my_param_name")
      {
        if (param.get_type() != rclcpp::ParameterType::PARAMETER_STRING)
        {
          result.successful = false;
          result.reason = "my_param_name must be a string";
          break;
        }

        // Optionally do something with parameter
      }
    }

    return result;
  }

  // Need to hold a pointer to the callback
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

```

# Rclcpp：TF2

TF2 库提供了对转换的轻松访问。以下所有示例都需要对 _tf2_ros_ 的依赖关系。

## 发布TF

```cpp
#include <tf2_ros/transform_broadcaster.h>
std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster;

broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(nodePtr);

geometry_msgs::msg::TransformStamped transform;
transform.header.stamp = node->now();
transform.header.frame_id = "odom";
transform.child_frame_id = "base_link";

// Fill in transform.transform.translation
// Fill in transform.transform.rotation

broadcaster->sendTransform(transform);
```

## 监听TF

```cpp
#include "tf2_ros/transform_listener.h"

std::shared_ptr<tf2_ros::Buffer> tf_buffer;
std::shared_ptr<tf2_ros::TransformListener> tf_listener;

rclcpp::Node node("name_of_node");

tf_buffer.reset(new tf2_ros::Buffer(node.get_clock()));
tf_listener.reset(new tf2_ros::TransformListener(*tf_buffer_));
```

## TF变换

TF2 可以通过提供实现的包进行扩展。GEOMETRY_msgs 程序包为 msgs_ 提供这些。下面的示例使用 msgs::msg::PointStamed_，但这应该适用于 msgs_ 中的任何数据类型：

```cpp
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

geometry_msg::msgs::PointStamped in, out;
in.header.frame_id = "source_frame";

try
{
  tf_buffer->transform(in, out, "target_frame");
}
catch (const tf2::TransformException& ex)
{
  RCLCPP_ERROR(rclcpp::get_logger("logger_name"), "Could not transform point.");
}
```

_transform_ 函数还可以接受超时。然后它将等待转换可用的时间达到这个时间量:

```cpp
tf_buffer->transform(in, out, "target_frame", tf2::durationFromSec(1.0));
```

## 获取最新TF

常见的工作方式是获得“最新”转换。在 ros2中，这可以使用 tf2::TimePointZero 来实现，但是需要使用 lookupTransform 然后调用 doTransform (基本上就是在内部进行转换) :

```cpp
geometry_msgs::msg::PointStamped in, out;

geometry_msgs::msg::TransformStamped transform =
   tf_buffer->lookupTransform("target_frame",
                              in.header.frame_id,
                              tf2::TimePointZero);

tf2::doTransform(in, out, transform);
```

# rclcpp: Time

_rclcpp::Time_ 和  _rclcpp::Duration_ 和ROS1中的用法偏差较大，但与[std::chrono](https://en.cppreference.com/w/cpp/chrono)
的关系更为密切。[ROS Discourse](https://discourse.ros.org/t/ros-2-time-vs-std-chrono/6293) 可以看到与其有关的比较深入的讨论。

在移植某些ros1库时，时间戳可能会被大量用作浮点秒。从 rclcpp 获取浮点秒 _rclcpp::Time_:

```cpp
// node is instance of rclcpp::Node
rclcpp::Time t = node.now();
double seconds = t.seconds();
```

没有用于从浮点表示的秒开始的时间的构造函数，因此你首先需要转换为纳秒：

```cpp
rclcpp::Time t(static_cast<uin64_t>(seconds * 1e9));
```

确实具有双向功能：

```cpp
rclcpp::Duration d = rclcpp::Duration::from_seconds(1.0);
double seconds = d.seconds();
```

# rclcpp: Point Clouds

`sensor_msgs/PointCloud2`  是一种非常常见的 ROS 消息类型，用于处理 ROS 中的感知数据。这也是实际要解释的最复杂的信息之一。

消息的复杂性源于它在单个巨型数据存储中包含任意字段这一事实。这允许 PointCloud2 消息与任何类型的云(例如，仅 XYZ 点、XYZRGB，甚至 XYZI)一起工作，但在访问云中的数据时增加了一些复杂性。

在 ROS1 中，有一个更简单的 PointCloud 消息，但这已经被删除，并将在 ROS2-G 中删除。

## 使用 PointCloud2 迭代器

sensor_msgs 包提供了一个 C++ PointCloud2Iterator，可用于创建、修改和访问 PointCloud2 消息。

要创建新消息：

```cpp
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

sensor_msgs::msg::PointCloud2 msg;

// Fill in the size of the cloud
msg.height = 480;
msg.width = 640;

// Create the modifier to setup the fields and memory
sensor_msgs::PointCloud2Modifier mod(msg);

// Set the fields that our cloud will have
mod.setPointCloud2FieldsByString(2, "xyz", "rgb");

// Set up memory for our points
mod.resize(msg.height * msg.width);

// Now create iterators for fields
sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");

for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b)
{
  *iter_x = 0.0;
  *iter_y = 0.0;
  *iter_z = 0.0;
  *iter_r = 0;
  *iter_g = 255;
  *iter_b = 0;
}
```

## 使用 PCL

对于许多操作，你可能希望转换为 pcl::PointCloud 以便使用的扩展 API [Point Cloud Library](https://pointclouds.org)。

在 ROS1 ，pcl_ros 包允许你编写一个订阅者，它的回调直接接受 pcl::PointCloud，但是这个包还没有被移植到 ROS2. 无论如何，使用 pcl_conversions 包自己进行转换是非常简单的：

```cpp
#include "pcl_conversions/pcl_conversions.h"

void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // PCL still uses boost::shared_ptr internally
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
    boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  // This will convert the message into a pcl::PointCloud
  pcl::fromROSMsg(*msg, *cloud);
}
```

反之，你也可以反方向转换：

```cpp
#include "pcl_conversions/pcl_conversions.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
sensor_msgs::msg::PointCloud2 msg;

pcl::toROSMsg(*cloud, msg);
cloud_publisher->publish(msg);
```

# rclcpp: Workarounds

## 懒订阅

ROS2 还没有订阅连接回叫。此代码创建一个函数，定期调用该函数来检查我们是否需要启动或停止订阅者：

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

class LazyPublisherEx : rclcpp::Node
{
public:
  LazyPublisherEx(const rclcpp::NodeOptions & options) :
    Node("lazy_ex", options)
  {
    // Setup timer
    timer = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&LazyPublisher::periodic, this));
  }

private:
  void periodic()
  {
    if (pub_.get_subscription_count() > 0)
    {
        // We have a subscriber, do any setup required
    }
    else
    {
        // Subscriber has disconnected, do any shutdown
    }
  }

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
```

在使用图像传输时也可以这样做，你只需要将 _get_subscription_count_ 更改为 _getNumSubscribers_:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.h>

class LazyPublisherEx : rclcpp::Node
{
public:
  LazyPublisherEx(const rclcpp::NodeOptions & options) :
    Node("lazy_ex", options)
  {
    // Setup timer
    timer = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&LazyPublisher::periodic, this));
  }

private:
  void periodic()
  {
    if (pub_.getNumSubscribers() > 0)
    {
        // We have a subscriber, do any setup required
    }
    else
    {
        // Subscriber has disconnected, do any shutdown
    }
  }

  image_transport::CameraPublisher pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
```
###### datetime:2023/10/24 10:23

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

## 1.Micro-ROS介绍与服务安装

本节我们主要介绍下Micro-ROS几大主要特点。

![](imgs/micro-ROS_architecture.png)

先上系统框架图，下面再一一介绍。

## 一、特点1：运行在微控制器上的ROS2

首先从名称看，Micro-ROS，Micro指的就是`microcontrollers`即微控制器。

![](imgs/image-20230121000431421.png)

核心作用就是上面这句话`micro-ROS puts ROS 2 onto microcontrollers`。既然是在微控制器上，因硬件资源受限，其功能肯定会有所裁剪，但核心的ROS2通信功能依然保有。

![](imgs/image-20230121002536573.png)

## 二、特点2：MicroROS支持多种通信协议并依赖Agent

![](imgs/image-20230121002714689.png)

所谓Agen其实就是一个代理，**微控制器可以通过串口，蓝牙、以太网、Wifi等多种协议将数据传递给Agent**，Agent再将其转换成ROS2的话题等数据，以此完成通信。

## 三、特点3：通过RCLC-API调用MicroROS

![](imgs/image-20230121003129698.png)

因为MicroROS遵循RCLCAPI，所以和在上位机中使用Python或者C++调用MicroROS有所不同，最终代码风格如下面这段所示

```c++
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;


void setup() {
  // Configure serial transport
  Serial.begin(115200);
  set_microros_serial_transports(Serial);
  delay(2000);

  allocator = rcl_get_default_allocator();

  //create init_options
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

  // create node
  RCCHECK(rclc_node_init_default(&node, "micro_ros_platformio_node", "", &support));

  // create publisher
  RCCHECK(rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
    "micro_ros_platformio_node_publisher"));


  // create executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  msg.data = 0;
}

```

## 四、在上位机上安装Agent

我们使用Docker来进行Agent的安装。

### 4.1 安装Docker

打开终端，复制粘贴输入下面代码

```
wget http://fishros.com/install -O fishros && . fishros
```

接着输入密码，在下面的界面输入8,一键安装Docker,完成后等待即可。

![](imgs/1663861023833-528a2dc4-de20-4b24-89eb-d9fe7b5b107d-image-16742326632143.png)

### 4.2 运行Agent

安装完成Docker后打开终端，输入下面的指令

```
sudo docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO serial --dev /dev/ttyUSB0 -v6
```

稍微等待下载完成，看到如下界面表示成功启动。

![](imgs/image-20230121004045577.png)

上面的指令是使用串口通讯协议运行microros-agent，还可以通过UDP、TCP、CAN等协议运行，具体指令如下

```shell
# UDPv4 micro-ROS Agent
docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO udp4 --port 8888 -v6

# Serial micro-ROS Agent
docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO serial --dev [YOUR BOARD PORT] -v6

# TCPv4 micro-ROS Agent
docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO tcp4 --port 8888 -v6

# CAN-FD micro-ROS Agent
docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO canfd --dev [YOUR CAN INTERFACE] -v6
```

## 五、总结

本节我们主要介绍了MicroROS的主要特点，接着介绍使用Docker下载和运行Agent，既然搞定了上位机，下一节我们正是开始在开发板上编写MicroROS节点，然后测试与上位机的连接是否正常。
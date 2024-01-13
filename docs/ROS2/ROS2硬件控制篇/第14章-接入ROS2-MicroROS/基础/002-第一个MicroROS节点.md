###### datetime:2023/10/24 10:23

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 2.你的第一个MicroROS节点

上一节我们介绍了MicroROS和Agent的安装，本节我们开始正是编写代码，接入ROS2。

## 一、新建工程添加依赖

### 1.1 新建工程

新建`example10_hello_microros`工程，这里需要更改下工程的位置，默认目录是在文档目录下，在测试时发现目录定位上有bug，所以建议建议直接放到主目录或其下目录，这里直接放到主目录。

![image-20230120232044724](imgs/image-20230120232044724-16742280577371.png)

### 1.2 添加依赖

打开`platform.ini`,接着我们添加MicroROS的依赖。

```ini
[env:featheresp32]
platform = espressif32
board = featheresp32
framework = arduino
lib_deps =
    https://gitee.com/ohhuo/micro_ros_platformio.git
```

这里使用的地址并不是MicroROS官方仓库，而是经过修改后的国内仓库地址，里面放置了编译好后可以直接使用的microros静态库，并对仓库中需要梯子的地址进行了替换。

## 二、编写代码-第一个节点

开始编写代码，因为Micro-ROS遵循RCLC-API，所以这里通过一个最简单的例程介绍如何新建一个节点。

```c++
#include <Arduino.h>
#include <micro_ros_platformio.h>

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

void setup()
{
  Serial.begin(115200);
  // 设置通过串口进行MicroROS通信
  set_microros_serial_transports(Serial);
  // 延时时一段时间，等待设置完成
  delay(2000);
  // 初始化内存分配器
  allocator = rcl_get_default_allocator();
  // 创建初始化选项
  rclc_support_init(&support, 0, NULL, &allocator);
  // 创建节点 hello_microros
  rclc_node_init_default(&node, "hello_microros", "", &support);
  // 创建执行器
  rclc_executor_init(&executor, &support.context, 1, &allocator);
}

void loop()
{
  delay(100);
  // 循环处理数据
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
}

```

上面代码并不复杂，已经将注释写上，强烈建议你跟着代码敲一遍，不要直接复制粘贴。

相比在上位机中开发ROS，这里多了几步

- 设置通信协议，因为可以通过多种方式连接，所以需要进行提前设置
- 初始化内存分配器，在微控制器上资源受限，内存的管理要很细致
- 创建初始化选项，用于初始化rcl并创建一些需要用到的数据结构体

关于rclc的api并没有找到文档，不过源码的头文件依然非常清晰，直接安装Ctrl点击某个函数即可跳转（不行的，重启下Vscode）。

比如关于rclc_support_init 的源码及参数介绍。

```c++
/**
 *  Initializes rcl and creates some support data structures.
 *  Initializes clock as RCL_STEADY_TIME.
 *  * <hr>
 * Attribute          | Adherence
 * ------------------ | -------------
 * Allocates Memory   | Yes (in RCL)
 * Thread-Safe        | No
 * Uses Atomics       | No
 * Lock-Free          | Yes
 *
 * \param[inout] support a zero-initialized rclc_support_t
 * \param[in] argc number of args of main
 * \param[in] argv array of arguments of main
 * \param[in] allocator allocator for allocating memory
 * \return `RCL_RET_OK` if RCL was initialized successfully
 * \return `RCL_RET_INVALID_ARGUMENT` if any null pointer as argument
 * \return `RCL_RET_ERROR` in case of failure
 */
RCLC_PUBLIC
rcl_ret_t
rclc_support_init(
  rclc_support_t * support,
  int argc,
  char const * const * argv,
  rcl_allocator_t * allocator);
```

## 三、运行测试

连接开发板，编译下载，如果遇到端口被占用，多半是你的microros_agent没有关闭，Ctrl+C打断运行再次尝试。

![](imgs/image-20230121011234354.png)

接着打开Agent

![](imgs/image-20230121011320762.png)

然而并没有什么反应，重新点击一次RST即可看到有数据发送和接收过来了。

![](imgs/image-20230121011410538.png)

接着打开新的终端，输入指令

```shell
ros2 node list
ros2 node info /hello_microros
```

![](imgs/image-20230121011552866.png)

可以看到，我们的第一个节点成功运行起来了。

## 四、总结

本节我们成功的在微控制器平台上将MicroROS节点运行起来了，下一节我们开始正式进行ROS2通信的学习。

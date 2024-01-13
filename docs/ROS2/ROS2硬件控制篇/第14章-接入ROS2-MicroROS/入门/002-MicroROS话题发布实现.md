###### datetime:2023/10/24 10:23

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 2.MicroROS-话题发布实现

本节将学习在开发板上实现话题的发布，最终实现通过话题发布当前开发板的电池电量信息，关于电量信息的测量，请参考：[电池电压测量-学会使用ADC](../../第13章-嵌入式开发之从点灯开始\入门\001-看懂LED驱动电路-GPIO控制-学会使用ADC.md)

## 一、新建工程添加依赖

新建`example12_microros_topic_pub`工程

![image-20230121022246635](imgs/image-20230121022246635.png)

修改`platformio.ini`添加依赖

```ini
; PlatformIO Project Configuration File
;
;   Build options: build flags, source filter
;   Upload options: custom upload port, speed and extra flags
;   Library options: dependencies, extra library storages
;   Advanced options: extra scripting
;
; Please visit documentation for the other options and examples
; https://docs.platformio.org/page/projectconf.html

[env:featheresp32]
platform = espressif32
board = featheresp32
framework = arduino
lib_deps =
    https://gitee.com/ohhuo/micro_ros_platformio.git
```

## 二、编写代码-实现订阅

编辑main.cpp，代码如下，注释已经添加到代码中来了

```c++
#include <Arduino.h>
#include <micro_ros_platformio.h>

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
// 添加头文件
#include <std_msgs/msg/float32.h>

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

// 声明话题发布者
rcl_publisher_t publisher;
// 声明消息文件
std_msgs__msg__Float32 pub_msg;

// 定义定时器接收回调函数
void timer_callback(rcl_timer_t *timer, int64_t last_call_time)
{
  RCLC_UNUSED(last_call_time);
  if (timer != NULL)
  {
    rcl_publish(&publisher, &pub_msg, NULL);
  }
}

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
  // 创建节点 topic_sub_test
  rclc_node_init_default(&node, "topic_pub_test", "", &support);
  // 订阅者初始化
  rclc_publisher_init_default(
      &publisher,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
      "battery_voltage");

  // 创建定时器，200ms发一次
  const unsigned int timer_timeout = 200;
  rclc_timer_init_default(
      &timer,
      &support,
      RCL_MS_TO_NS(timer_timeout),
      timer_callback);

  // 创建执行器
  rclc_executor_init(&executor, &support.context, 1, &allocator);
  // 给执行器添加定时器
  rclc_executor_add_timer(&executor, &timer);
  // 初始化ADC
  pinMode(34, INPUT);
  analogSetAttenuation(ADC_11db);
}

void loop()
{
  delay(100);
  // 循环处理数据
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
  // 通过ADC获取电压值
  int analogValue = analogRead(34);                     // 读取原始值0-4096
  int analogVolts = analogReadMilliVolts(34);           // 读取模拟电压，单位毫伏
  float realVolts = 5.02 * ((float)analogVolts * 1e-3); // 计算实际电压值
  pub_msg.data = realVolts;
}

```

## 三、代码注解

相比之前的节点代码这里主要多了这几行

- `#include <std_msgs/msg/float32.h>` 包含flaot32类型头文件
- `rcl_publisher_t publisher; 定义发布者`
- `std_msgs__msg__Float32 pub_msg; 定义发布消息，也需要提前定义`
- **`void timer_callback(rcl_timer_t *timer, int64_t last_call_time)` 定义定时器回调函数，当我们需要以某个频率做什么的时候定时器可以派上用场**
- **`rclc_publisher_init_default` 初始化发布者**
- **`rclc_timer_init_default 初始化定时器`**
- **`rclc_executor_add_timer 给执行器添加一个定时器回调`**

## 四、下载测试

### 4.1 编译下载

连接开发板，编译下载。

![image-20230121100425404](imgs/image-20230121100425404.png)

### 4.2 启动Agent服务

接着打开终端启动agent

```shell
sudo docker run -it --rm -v /dev:/dev -v /dev/shm:/dev/shm --privileged --net=host microros/micro-ros-agent:$ROS_DISTRO serial --dev /dev/ttyUSB0 -v
```

点击下RST按钮，重启开发板，正常可以看到下图内容

![image-20230121115751128](imgs/image-20230121115751128.png)

### 4.3 测试是否连通

```shell
ros2 node list
ros2 topic list
```

![image-20230121115828533](imgs/image-20230121115828533.png)

### 4.4 查看话题数据

```shell
ros2 topic echo /battery_voltage
```

![image-20230121115849458](imgs/image-20230121115849458.png)

这里连接了小车的电池，VM电压代表电池电压，符合正常电压值范围。

同时可以使用下面指令测量话题频率

```
fishros@fishros-MS-7D42:~/example12_microros_topic_pub$ ros2 topic hz /battery_voltage 
average rate: 4.828
        min: 0.207s max: 0.208s std dev: 0.00021s window: 6
average rate: 5.034
        min: 0.106s max: 0.208s std dev: 0.02793s window: 12
average rate: 4.973
        min: 0.106s max: 0.208s std dev: 0.02378s window: 17
average rate: 4.941
        min: 0.106s max: 0.208s std dev: 0.02104s window: 22
average rate: 5.005
        min: 0.106s max: 0.208s std dev: 0.02594s window: 28
average rate: 4.977
        min: 0.106s max: 0.208s std dev: 0.02404s window: 33
average rate: 4.958
        min: 0.106s max: 0.208s std dev: 0.02249s window: 38
average rate: 4.997
        min: 0.106s max: 0.208s std dev: 0.02541s window: 44
```

## 五、总结

本节我们通过电量信息发布例程，学习了如何在开发板上实现话题发布流程。下一节我们开始尝试在开发板上建立服务端，尝试服务通信。
###### datetime:2022/04/25 10:51

###### author:nzb

# rospy与主要接口

## rospy vs roscpp

rospy是Python版本的ROS客户端库，提供了Python编程需要的接口，你可以认为rospy就是一个Python的模块(Module)
。这个模块位于`/opt/ros/kineetic/lib/python2.7/dist-packages/rospy`之中。

rospy包含的功能与roscpp相似，都有关于node、topic、service、param、time相关的操作。但同时rospy和roscpp也有一些区别：

- rospy没有一个NodeHandle，像创建publisher、subscriber等操作都被直接封装成了rospy中的函数或类，调用起来简单直观。
- rospy一些接口的命名和roscpp不一致，有些地方需要开发者注意，避免调用错误。

相比于C++的开发，用Python来写ROS程序开发效率大大提高，诸如显示、类型转换等细节不再需要我们注意，节省时间。但Python的执行效率较低，同样一个功能用Python运行的耗时会高于C++。因此我们开发SLAM、路径规划、机器视觉等方面的算法时，往往优先选择C++。

ROS中绝大多数基本指令，例如`rostopic`,`roslaunch`都是用python开发的，简单轻巧。

## ROS中Python代码的组织方式

要介绍rospy，就不得不提Python代码在ROS中的组织方式。通常来说，Python代码有两种组织方式，一种是单独的一个Python脚本，适用于简单的程序，另一种是Python模块，适合体量较大的程序。

### 单独的Python脚本

对于一些小体量的ROS程序，一般就是一个Python文件，放在script/路径下，非常简单。

```text
your_package
|- script/
|- your_script.py
|-...
```

### Python模块

当程序的功能比较复杂，放在一个脚本里搞不定时，就需要把一些功能放到Python Module里，以便其他的脚本来调用。ROS建议我们按照以下规范来建立一个Python的模块：

```text
your_package
|- src/
|-your_package/
|- _init_.py
|- modulefiles.py
|- scripts/
|- your_script.py
|- setup.py
```

在src下建立一个与你的package同名的路径，其中存放`_init_.py`以及你的模块文件。这样就建立好了ROS规范的Python模块，你可以在你的脚本中调用。 如果你不了解`init.py`
的作用，可以参考这篇[博客](http://www.cnblogs.com/Lands-ljk/p/5880483.html)
ROS中的这种Python模块组织规范与标准的Python模块规范并不完全一致，你当然可以按照Python的标准去建立一个模块，然后在你的脚本中调用，但是我们还是建议按照ROS推荐的标准来写，这样方便别人去阅读。

通常我们常用的ROS命令，大多数其实都是一个个Python模块，源代码存放在[ros_comm](https://github.com/ros/ros_comm/tree/lunar-devel/tools)
仓库的tools路径下你可以看到每一个命令行工具（如rosbag、rosmsg）都是用模块的形式组织核心代码，然后在`script/`下建立一个脚本来调用模块。

## 常用rospy的API

这里分类整理了rospy常见的一些用法，请你浏览一遍，建立一个初步的影响。 具体[API](http://docs.ros.org/api/rospy/html/rospy-module.html) 请查看

### Node相关

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
| rospy.init_node(name, argv=None, anonymous=False) | 注册和初始化node |   |
| MasterProxy | rospy.get_master() | 获取master的句柄 |
| bool | rospy.is_shutdown() | 节点是否关闭 |
| rospy.on_shutdown(fn) |    在节点关闭时调用fn函数 |  |
| str | get_node_uri() | 返回节点的URI |
| str | get_name() | 返回本节点的全名 |
| str | get_namespace() | 返回本节点的名字空间 |
| ... | ... | ... |

### Topic相关

- 函数：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
| [[str, str]] | get_published_topics() | 返回正在被发布的所有topic名称和类型 |
| Message | wait_for_message(topic, topic_type, time_out=None) | 等待某个topic的message |
|  | spin() |    触发topic或service的回调/处理函数，会阻塞直到关闭节点 |
| ... | ... | ... |

- Publisher类：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
|   | \_\_init__(self, name, data_class, queue_size=None) |    构造函数 |
|   |  publish(self, msg) |    发布消息 |
| str | unregister(self) | 停止发布 |
| ... | ... | ... |

- Subscriber类：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
|   | \_\_init__(self, name, data_class, call_back=None, queue_size=None) | 构造函数 |
|   | unregister(self, msg) |    停止订阅 |
| ... | ... | ... |

### Service相关

- 函数：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
| |wait_for_service(service, timeout=None) |    阻塞直到服务可用 |
| ... | ... | ... |

- Service类(server)：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
|  | \_\_init__(self, name, service_class, handler) | 构造函数，handler为处理函数，service_class为srv类型 |
| | shutdown(self) | 关闭服务的server |
| ... | ... | ... |

- ServiceProxy类(client)：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
|  | \_\_init__(self, name, service_class) | 构造函数，创建client |
|  | call(self, args, *kwds) | 发起请求 |
|  | \_\_call__(self, args, *kwds) | 同上 |
| | close(self)     | 关闭服务的client |
| ... | ... | ... |

### Param相关

- 函数：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
| XmlRpcLegalValue | get_param(param_name, default=_unspecified) | 获取参数的值 |
| [str] | get_param_names() | 获取参数的名称 |
| | set_param(param_name, param_value) | 设置参数的值 |
| | delete_param(param_name) |    删除参数 |
| bool | has_param(param_name) | 参数是否存在于参数服务器上 |
| str | search_param() | 搜索参数 |
| ... | ... | ... |

### 时钟相关

- 函数：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
| Time | get_rostime() | 获取当前时刻的Time对象 |
| float | get_time() | 返回当前时间，单位秒 |
|  | sleep(duration) |    执行挂起 |
| ... | ... | ... |

- Time类：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
|  | \_\_init__(self, secs=0, nsecs=0) |    构造函数 |
| Time | now() | 静态方法 返回当前时刻的Time对象 |
| ... | ... | ... |

- Duration类：

| 返回值 | 方法 | 作用 |
| ----- | ----- | ----- |
|   | \_\_init__(self, secs=0, nsecs=0) | 构造函数 |
| ... | ... | ... |

# topic in rospy

与roscpp类似，我们用python来写一个节点间消息收发的demo，同样还是创建一个自定义的gps类型的消息，一个节点发布模拟的gps信息，另一个接收和计算距离原点的距离。

## 自定义消息的生成

gps.msg定义如下：

```text
string state   #工作状态
float32 x      #x坐标
float32 y      #y坐标
```

我们需要修改`CMakeLists.txt`文件，方法见5.3节，这里需要强调一点的就是，对创建的msg进行`catkin_make`
会在`~/catkin_ws/devel/lib/python2.7/dist-packages/topic_demo`下生成msg模块（module）。
有了这个模块，我们就可以在python程序中`from topic_demo.msg import gps`,从而进行gps类型消息的读写。

## 消息发布节点

与C++的写法类似，我们来看topic用Python如何编写程序，见`topic_demo/scripts/pytalker.py`：

```python
import rospy
# 导入自定义的数据类型
from topic_demo.msg import gps


def talker():
    # Publisher 函数第一个参数是话题名称，第二个参数 数据类型，现在就是我们定义的msg 最后一个是缓冲区的大小
    # queue_size: None（不建议）  #这将设置为阻塞式同步收发模式！
    # queue_size: 0（不建议）#这将设置为无限缓冲区模式，很危险！
    # queue_size: 10 or more  #一般情况下，设为10 。queue_size太大了会导致数据延迟不同步。
    pub = rospy.Publisher('gps_info', gps, queue_size=10)
    rospy.init_node('pytalker', anonymous=True)
    # 更新频率是1hz
    rate = rospy.Rate(1)
    x = 1.0
    y = 2.0
    state = 'working'
    while not rospy.is_shutdown():
        # 计算距离
        rospy.loginfo('Talker: GPS: x=%f ,y= %f', x, y)
        pub.publish(gps(state, x, y))
        x = 1.03 * x
        y = 1.01 * y
        rate.sleep()


if __name__ == '__main__':
    talker()

```

以上代码与C++的区别体现在这几个方面：

- rospy创建和初始化一个node，不再需要用NodeHandle。rospy中没有设计NodeHandle这个句柄，我们创建topic、service等等操作都直接用rospy里对应的方法就行。
- rospy中节点的初始化并一定得放在程序的开头，在Publisher建立后再初始化也没问题。
- 消息的创建更加简单，比如gps类型的消息可以直接用类似于构造函数的方式`gps(state,x,y)`来创建。
- 日志的输出方式不同，C++中是`ROS_INFO()`，而Python中是`rospy.loginfo()`
- 判断节点是否关闭的函数不同，C++用的是`ros::ok()`而Python中的接口是`rospy.is_shutdown()`

通过以上的区别可以看出，roscpp和rospy的接口并不一致，在名称上要尽量避免混用。在实现原理上，两套客户端库也有各自的实现，并没有基于一个统一的核心库来开发。这也是ROS在设计上不足的地方。

ROS2就解决了这个问题，ROS2中的客户端库包括了`rclcpp`(ROS Clinet Library C++)、`rclpy`(ROS Client Library Python)
,以及其他语言的版本，他们都是基于一个共同的核心ROS客户端库rcl来开发的，这个核心库由C语言实现。

## 消息订阅节点

见`topic_demo/scripts/pylistener.py`：

```python
import rospy
import math
# 导入mgs
from topic_demo.msg import gps


# 回调函数输入的应该是msg
def callback(gps):
    distance = math.sqrt(math.pow(gps.x, 2) + math.pow(gps.y, 2))
    rospy.loginfo('Listener: GPS: distance=%f, state=%s', distance, gps.state)


def listener():
    rospy.init_node('pylistener', anonymous=True)
    # Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
    rospy.Subscriber('gps_info', gps, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
```

在订阅节点的代码里，rospy与roscpp有一个不同的地方：rospy里没有`spinOnce()`，只有`spin()`。

建立完talker和listener之后，经过`catkin_make`，就完成了python版的topic通信模型。

# Service in rospy

本节用python来写一个节点间，利用Service通信的demo，与5.4类似，创建一个节点，发布模拟的gps信息，另一个接收和计算距离原点的距离。

## srv文件

在5.4节，我们已经说过要建立一个名为`Greeting.srv`的服务文件，内容如下：

```text
string name #短横线上边部分是服务请求的数据
int32 age
--- #短横线下面是服务回传的内容
string feedback
```

然后修改`CMakeLists.txt`
文件。ROS的catkin编译系统会将你自定义的msg、srv（甚至还有action）文件自动编译构建，生成对应的C++、Python、LISP等语言下可用的库或模块。许多初学者错误地以为，只要建立了一个msg或srv文件，就可以直接在程序中使用，这是不对的，必须在`CMakeLists.txt`
中添加关于消息创建、指定消息/服务文件那几个宏命令。

## 创建提供服务节点(server)

见`service_demo/scripts/server_demo.py`：

```python
import rospy
from service_demo.srv import *


def server_srv():
    # 初始化节点，命名为 "greetings_server"
    rospy.init_node("greetings_server")
    # 定义service的server端，service名称为"greetings"， service类型为Greeting
    # 收到的request请求信息将作为参数传递给handle_function进行处理
    s = rospy.Service("greetings", Greeting, handle_function)
    rospy.loginfo("Ready to handle the request:")
    # 阻塞程序结束
    rospy.spin()


def handle_function(req):
    # 注意我们是如何调用request请求内容的，是将其认为是一个对象的属性，在我们定义
    # 的Service_demo类型的service中，request部分的内容包含两个变量，一个是字符串类型的name，另外一个是整数类型的age
    rospy.loginfo('Request from %s with age %d', req.name, req.age)
    # 返回一个Service_demo.Response实例化对象，其实就是返回一个response的对象，其包含的内容为我们在Service_demo.srv中定义的
    # response部分的内容，我们定义了一个string类型的变量feedback，因此，此处实例化时传入字符串即可
    return GreetingResponse("Hi %s. I' server!" % req.name)


# 如果单独运行此文件，则将上面定义的server_srv作为主函数运行
if __name__ == "__main__":
    server_srv()
```

以上代码中可以看出Python和C++在ROS服务通信时，server端的处理函数有区别： C++的handle_function()
传入的参数是整个srv对象的request和response两部分，返回值是bool型，显示这次服务是否成功的处理，也就是：

```text
bool handle_function(service_demo::Greeting::Request &req, service_demo::Greeting::Response &res){
...
    return true;
}
```

而Python的handle_function()传入的只有request，返回值是response，即：

```text
def handle_function(req):
    ...
    return GreetingResponse("Hi %s. I' server!"%req.name)
```

这也是ROS在两种语言编程时的差异之一。相比来说Python的这种思维方式更加简单，符合我们的思维习惯。

## 创建服务请求节点(client)

`service_demo/srv/client.cpp`内容如下：

```python
import rospy
from service_demo.srv import *


def client_srv():
    rospy.init_node('greetings_client')
    # 等待有可用的服务 "greetings"
    rospy.wait_for_service("greetings")
    try:
        # 定义service客户端，service名称为“greetings”，service类型为Greeting
        greetings_client = rospy.ServiceProxy("greetings", Greeting)

        # 向server端发送请求，发送的request内容为name和age,其值分别为"HAN", 20
        # 此处发送的request内容与srv文件中定义的request部分的属性是一致的
        # resp = greetings_client("HAN",20)
        resp = greetings_client.call("HAN", 20)
        rospy.loginfo("Message From server:%s" % resp.feedback)
    except rospy.ServiceException, e:
        rospy.logwarn("Service call failed: %s" % e)


# 如果单独运行此文件，则将上面函数client_srv()作为主函数运行
if __name__ == "__main__":
    client_srv()
```

以上代码中`greetings_client.call("HAN",20)`等同于`greetings_client("HAN",20)`。

# param与time

## param_demo

相比roscpp中有两套对param操作的API，rospy关于param的函数就显得简单多了，包括了增删查改等用法：

- rospy.get_param()
- rospy.set_param()
- rospy.has_param()
- rospy.delete_param()
- rospy.search_param()
- rospy.get_param_names()

下面我们来看看param_demo里的代码：

```python
import rospy


def param_demo():
    rospy.init_node("param_demo")
    rate = rospy.Rate(1)
    while (not rospy.is_shutdown()):
        # get param
        parameter1 = rospy.get_param("/param1")
        parameter2 = rospy.get_param("/param2", default=222)
        rospy.loginfo('Get param1 = %d', parameter1)
        rospy.loginfo('Get param2 = %d', parameter2)

        # delete param
        rospy.delete_param('/param2')

        # set param
        rospy.set_param('/param2', 2)

        # check param
        ifparam3 = rospy.has_param('/param3')
        if (ifparam3):
            rospy.loginfo('/param3 exists')
        else:
            rospy.loginfo('/param3 does not exist')

        # get all param names
        params = rospy.get_param_names()
        rospy.loginfo('param list: %s', params)

        rate.sleep()


if __name__ == "__main__":
    param_demo()
```

## time_demo

### 时钟

rospy中的关于时钟的操作和roscpp是一致的，都有Time、Duration和Rate三个类。 首先，Time和Duration前者标识的是某个时刻（例如今天22:00），而Duration表示的是时长(例如一周)
。但他们具有相同的结构（秒和纳秒）：

```text
int32 secs
int32 secs
```

### 创建Time和Duration

rospy中的Time和Duration的构造函数类似，都是`_init_(self,secs=0, nsecs=0)`,指定秒和纳秒(1ns = 10^-9 s)

```python
time_now1 = rospy.get_rostime()  # 当前时刻的Time对象 返回Time对象
time_now2 = rospy.Time.now()  # 同上
time_now3 = rospy.get_time()  # 得到当前时间，返回float 4单位秒
time_4 = rospy.Time(5)  # 创建5s的时刻
duration = rospy.Duration(3 * 60)  # 创建3min时长
```

关于Time、Duration之间的加减法和类型转换，和roscpp中的完全一致，请参考5.6节，此处不再重复。

### sleep

```python
duration.sleep()  # 挂起
rospy.sleep(duration)  # 同上，这两种方式效果完全一致

loop_rate = Rate(5)  # 利用Rate来控制循环频率
while (rospy.is_shutdown()):
    loop_rate.sleep()  # 挂起，会考虑上次loop_rate.sleep的时间
```

关于sleep的方法，Rate类中的sleep主要用来保持一个循环按照固定的频率，循环中一般都是发布消息、执行周期性任务的操作。这里的sleep会考虑上次sleep的时间，从而使整个循环严格按照指定的频率。

### 定时器Timer

rospy里的定时器和roscpp中的也类似，只不过不是用句柄来创建，而是直接`rospy.Timer(Duration, callback)`，第一个参数是时长，第二个参数是回调函数。

```python
def my_callback(event):
    print
    'Timer called at ' + str(event.current_real)


rospy.Timer(rospy.Duration(2), my_callback)  # 每2s触发一次callback函数
rospy.spin()
```

同样不要忘了`rospy.spin()`，只有spin才能触发回调函数。 回调函数的传入值是`TimerEvent`类型，该类型包括以下几个属性：

```text
rospy.TimerEvent
    last_expected
    理想情况下为上一次回调应该发生的时间
    last_real
    上次回调实际发生的时间
    current_expected
    本次回调应该发生的时间
    current_real
    本次回调实际发生的时间
    last_duration
    上次回调所用的时间（结束-开始）
```

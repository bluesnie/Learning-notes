###### datetime:2024/06/15 13:35

###### author:nzb

# lcm 

[官方文档](https://lcm-proj.github.io/lcm/content/build-instructions.html)

`LCM（Lightweight Commuciation and Marshalling）` 它是作为消息传递和封装的通信库，其首要任务是简化低时延消息传递系统的开发。目前广泛应用于无人驾驶汽车领域。

机器人通讯中有许多流行的通讯中间件，如百度`Apollo`的`Cyber RT`，`ROS1`中的`TCPROS/UDPROS`通信机制，`ROS2`中使用的`DDS`等等和`IPC( Inter-Process Communication` 系统自带的进程间通信)相比，也要高一些。

在机器人和自动驾驶系统中，`LCM`可以作为`ROS`的替代品，用于完成进程间、设备间的通讯。

LCM具有如下特性：

- 低延迟的进程间通信
- 使用UDP组播的高效广播机制
- 类型安全的消息编排
- 用户友好的记录和回放工具
- 没有集中的 "数据库 "或 “枢纽”–节点间直接通讯
- 没有守护进程
- 极少的依赖

## 数据类型

### 原始类型

| 名称 | 描述 |
| --- | --- |
| int8_t | 8位有符号整数 |
| int16_t | 16位有符号整数 |
| int32_t | 32位有符号整数 |
| int64_t | 64位有符号整数 |
| float | 32 位 IEEE 浮点数值 |
| double | 64 位 IEEE 浮点数值 |
| string | UTF-8字符串 |
| boolean | true/false  |
| byte | 8位数据 |


### 数组

```cpp
struct point2d_list_t
{
    int32_t npoints;
    double  points[npoints][2];
}
// 可变长度的二维数组
// points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1]
```

### 常量

```cpp
struct my_constants_t
{
    const int32_t YELLOW=1, GOLDENROD=2, CANARY=3;
    const double E=2.8718;
}

// 字符串常量不支持
```
### 命名空间（包）

```cpp
package mycorp;

struct camera_image_t {
    int64_t      utime;
    string       camera_name;
    jpeg.image_t jpeg_image;
    mit.pose_t   pose;
}
```

## LCM-gen

### 示例`example_t.lcm`

```cpp
package exlcm;

struct example_t
{
    int64_t  timestamp;
    double   position[3];
    double   orientation[4]; 
    int32_t  num_ranges;
    int16_t  ranges[num_ranges];
    string   name;
    boolean  enabled;
}
```

### 生成特定的语言文件

| 语言 | 命令 |
| --- | --- |
| C | `lcm-gen -c example_t.lcm` |
| C++ | `lcm-gen -x example_t.lcm` |
| Java | `lcm-gen -j example_t.lcm` |
| Lua | `lcm-gen -l example_t.lcm` |
| Python | `lcm-gen -p example_t.lcm` |
| C# | `lcm-gen -csharp example_t.lcm` |
| MATLAB | `Generate Java code` |
| Go | `lcm-gen -g example_t.lcm` |


## CPP示例

- `publisher.cpp`

```cpp
#include <lcm/lcm-cpp.hpp>
#include "exlcm/example_t.hpp"
/*
compile:  g++ -I /home/blues/cyan_manipulation/install/lcm/include/ -L /home/blues/cyan_manipulation/install/lcm/lib publish.cpp -o publish -llcm
run: 
    export LD_LIBRARY_PATH=/home/blues/cyan_manipulation/install/lcm/lib:$LD_LIBRARY_PATH
    ./publish
*/
int main(int argc, char ** argv)
{
    lcm::LCM lcm;
    if(!lcm.good())
        return 1;

    exlcm::example_t my_data;
    my_data.timestamp = 0;

    my_data.position[0] = 1;
    my_data.position[1] = 2;
    my_data.position[2] = 3;

    my_data.orientation[0] = 1;
    my_data.orientation[1] = 0;
    my_data.orientation[2] = 0;
    my_data.orientation[3] = 0;

    my_data.num_ranges = 15;
    my_data.ranges.resize(my_data.num_ranges);
    for(int i = 0; i < my_data.num_ranges; i++)
        my_data.ranges[i] = i;

    my_data.name = "example string";
    my_data.enabled = true;

    lcm.publish("EXAMPLE", &my_data);

    return 0;
}
```

- `subscriber.py`

```cpp
#include <stdio.h>
#include <lcm/lcm-cpp.hpp>
#include "exlcm/example_t.hpp"

/*
compile:  g++ -I /home/blues/cyan_manipulation/install/lcm/include/ -L /home/blues/cyan_manipulation/install/lcm/lib subscribe.cpp -o subscribe -llcm
run: 
    - export LD_LIBRARY_PATH=/home/blues/cyan_manipulation/install/lcm/lib:$LD_LIBRARY_PATH
    - ./subscribe
*/

class Handler 
{
    public:
        ~Handler() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const exlcm::example_t* msg)
        {
            int i;
            printf("Received message on channel \"%s\":\n", chan.c_str());
            printf("  timestamp   = %lld\n", (long long)msg->timestamp);
            printf("  position    = (%f, %f, %f)\n",
                    msg->position[0], msg->position[1], msg->position[2]);
            printf("  orientation = (%f, %f, %f, %f)\n",
                    msg->orientation[0], msg->orientation[1], 
                    msg->orientation[2], msg->orientation[3]);
            printf("  ranges:");
            for(i = 0; i < msg->num_ranges; i++)
                printf(" %d", msg->ranges[i]);
            printf("\n");
            printf("  name        = '%s'\n", msg->name.c_str());
            printf("  enabled     = %d\n", msg->enabled);
        }
};

int main(int argc, char** argv)
{
    lcm::LCM lcm;
    if(!lcm.good())
        return 1;

    Handler handlerObject;
    lcm.subscribe("EXAMPLE", &Handler::handleMessage, &handlerObject);

    while(0 == lcm.handle());

    return 0;
}
```

## Python示例

- `publisher.py`

```python
import lcm
from exlcm import example_t

msg = example_t()
msg.timestamp = 0
msg.position = (1, 2, 3)
msg.orientation = (1, 0, 0, 0)
msg.ranges = range(15)
msg.num_ranges = len(msg.ranges)
msg.name = "example string"
msg.enabled = True

lc = lcm.LCM()
lc.publish("EXAMPLE", msg.encode())
```

- `subscriber.py`

```python
import lcm
from exlcm import example_t

def my_handler(channel, data):
    msg = example_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   position    = %s" % str(msg.position))
    print("   orientation = %s" % str(msg.orientation))
    print("   ranges: %s" % str(msg.ranges))
    print("   name        = '%s'" % msg.name)
    print("   enabled     = %s" % str(msg.enabled))
    print("")

lc = lcm.LCM()
subscription = lc.subscribe("EXAMPLE", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass
```

## API

- [cpp](https://lcm-proj.github.io/lcm/doxygen_output/c_cpp/html/group__LcmCpp.html)
- [python](https://lcm-proj.github.io/lcm/python/index.html)

## 教程和示例

[各个语言的教程和示例链接](https://lcm-proj.github.io/lcm/content/tutorial.html#)

## 可视化

- `make_types.sh`

```shell
#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN} Starting LCM type generation...${NC}"

cd ../jar_data  # 进到需要生成jar包路径

# Clean
rm -rf */*.jar
rm -rf */*.java
rm -rf */*.class

# Make
# 生成java文件 -j
lcm-gen -j ../lcm_types/common/**.lcm 
lcm-gen -j ../lcm_types/lowlevel_sdk/**.lcm
lcm-gen -j ../lcm_types/manipulation/**.lcm
cp /usr/local/share/java/lcm.jar .
javac -cp lcm.jar */*.java
jar cf manipulation_types.jar */*.class
mkdir -p jar_java
mv lcm.jar manipulation_types.jar jar_java  # 生成自定义jar包
export CLASSPATH=${DIR}/../jar_data/jar_java/manipulation_types.jar

echo -e "${GREEN} Done with LCM type generation${NC}"
```


- `launch_lcm_spy.sh`

```shell
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/../jar_data/jar_java
export CLASSPATH=${DIR}/../jar_data/jar_java/manipulation_types.jar  # 设置环境变量
pwd
lcm-spy
```



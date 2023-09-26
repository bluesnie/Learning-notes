###### datetime:2023/09/19 14:11

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 1.启动管理工具-Launch

## 1.Launch启动工具介绍

### 1.1 问题描述

对于一个机器人系统来说，往往由很多个不同功能的节点组成，启动一个机器人系统时往往需要启动多个节点，同时根据应用场景和机器人的不同，每个节点还会有不同的配置项。

如果每个节点我们都开一个新终端，敲`ros2 run`指令并写一堆参数，这是多么浪费生命且令人绝望的事情。

除了启动，你会发现，一个个关闭也是很难受的。

### 1.2 解决方案

可不可以编写一个类似于脚本的文件来管理节点的启动呢？

ROS2设计时就为我们想好了，为我们设计了一套完整的语法和规则的文件来帮助我们组织节点的启动，这个武器就叫launch文件。

**launch文件允许我们同时启动和配置多个包含 ROS 2 节点的可执行文件**

> 在ROS1中launch文件只有一种格式以.launch结尾的xml文档，不熟悉的同学写起来被xml语法折磨的死去活来。不过在ROS2中不要担心，因为在ROS2你可以使用Python代码来编写launch文件

## 2.编写第一个ROS2的launch文件

### 2.1 三种编写launch文件的方法

ROS2的launch文件有三种格式，python、xml、yaml。其中ROS2官方推荐的时python方式编写launch文件。 原因在于，相较于XML和YAML，**
Python是一个编程语言，更加的灵活，我们可以利用Python的很多库来做一些其他工作**（比如创建一些初始化的目录等）。

> 除了灵活还有另外一个原因是ros2/launch（一般launch共功能）和ros2/launch_ros（ROS 2 launch的特性）是用 Python 编写的，我们使用python编写launch文件可以使用 XML 和 YAML 中不能用的launch功能。
> 要说使用python版本的launch有什么坏处，那就是写起来比yaml要冗余

### 2.2 使用Python编写Launch

我们的目标是编写一个launch文件，最后使用launch指令，同时启动服务端和客户端节点。

#### 2.2.1 创建功能包和launch文件

创建文件夹和功能包，接着touch一个launch文件，后缀为`.py`。

```
mkdir -p chapt5/chapt5_ws/src
cd chapt5/chapt5_ws/src
ros2 pkg create robot_startup --build-type ament_cmake --destination-directory src
mkdir -p src/robot_startup/launch
touch src/robot_startup/launch/example_action.launch.py
```

#### 2.2.2 启动多个节点的示例

我们需要导入两个库，一个叫做LaunchDescription，用于对launch文件内容进行描述，一个是Node，用于声明节点所在的位置。

> 注意这里要定一个名字叫做`generate_launch_description`的函数，ROS2会对该函数名字做识别。

```python
# 导入库
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """launch内容描述函数，由ros2 launch 扫描调用"""
    action_robot_01 = Node(
        package="example_action_rclcpp",
        executable="action_robot_01"
    )
    action_control_01 = Node(
        package="example_action_rclcpp",
        executable="action_control_01"
    )
    # 创建LaunchDescription对象launch_description,用于描述launch文件
    launch_description = LaunchDescription(
        [action_robot_01, action_control_01])
    # 返回让ROS2根据launch描述执行节点
    return launch_description

```

#### 2.2.3 将launch文件拷贝到安装目录

如果你编写完成后直接编译你会发现install目录下根本没有你编写的launch文件，后续launch自然也找不到这个launch文件。

因为我们用的是`ament_cmake`类型功能包，所以这里要使用cmake命令进行文件的拷贝

```cmake
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
```

如果是`ament_python`功能包版

```python
from setuptools import setup
from glob import glob
import os

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
},
)

```

#### 2.2.4 编译测试

使用colcon指令编译我们的程序

```shell
colcon build
```

编译完成后，在`chapt5/chapt5_ws/install/robot_startup/share/robot_startup/launch`目录下你应该就可以看到被cmake拷贝过去的launch文件了。

接着运行

```sheel
# source 第五章的工作目录，这样才能找到对应的节点，不信你可以不source试试
source ../../chapt5/chapt5_ws/install/setup.bash
source install/setup.bash
ros2 launch robot_startup example_action.launch.py
# 新终端
ros2 node list #即可看到两个节点
```

![image-20220616135356671](imgs/image-20220616135356671.png)

## 3 添加参数&修改命名空间

接着我们尝试使用launch运行参数节点，并通过launch传递参数，和给节点以不同的命名空间。

新建`chapt5/chapt5_ws/src/robot_startup/launch/example_param_rclcpp.launch.py`。

编写内容如下

```python
# 导入库
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """launch内容描述函数，由ros2 launch 扫描调用"""
    parameters_basic1 = Node(
        package="example_parameters_rclcpp",
        namespace="rclcpp",
        executable="parameters_basic",
        parameters=[{'rcl_log_level': 40}]
    )
    parameters_basic2 = Node(
        package="example_parameters_rclpy",
        namespace="rclpy",
        executable="parameters_basic",
        parameters=[{'rcl_log_level': 50}]
    )
    # 创建LaunchDescription对象launch_description,用于描述launch文件
    launch_description = LaunchDescription(
        [parameters_basic1, parameters_basic2])
    # 返回让ROS2根据launch描述执行节点
    return launch_description
```

编译运行测试

```shell
# source 第五章的工作目录，这样才能找到对应的节点，不信你可以不source试试
source ../../chapt5/chapt5_ws/install/setup.bash
source install/setup.bash
ros2 launch robot_startup example_param_rclcpp.launch.py
# 新终端
ros2 node list #即可看到两个节点
```

![image-20220616140109862](imgs/image-20220616140109862.png)

## 4. 深入了解

### 如何编写launch文件

在ROS1中launch文件是一种格式以`.launch`结尾的xml文档；而在ROS2中推荐使用Python方式编写launch文件，此时的launch文件是一种格式以`.launch.py`结尾的Python脚本。

#### 启动节点

对于一个基础的启动节点的`launch`文件，需要引用以下库，然后创建一个名为做`generate_launch_description`的函数

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ......
```

**下文中未注明的均在generate_launch_description()函数中进行操作。**

创建`LaunchDescription`的对象`ld`（名字任意）

```python
ld = LaunchDescription()
```

然后创建一个`Actions.Node`对象

```python
example - node = Node(
    package='package-name',  # 节点所在的功能包
    namespace='package-namespace',  # 命名空间。如果存在同名节点，这一选项会有用
    executable='execute-name/script-name.py',  # 表示要运行的可执行文件名或脚本名字.py
    parameters=[{'parameter-name': parameter - value}],  # 参数
    arguments=['-xxx', xxx, '-xxx', xxx],  # 启动参数
    output='screen',  # 用于将话题信息打印到屏幕
    name='node-name'  # 表示启动后的节点名，可以没有
remappings = [  # 重映射
    ('/xxx/xxx-new', '/xxx/xxx-old'),
]
)
example - node2 = Node(
    ......
)
```

将上面所有的`Node`对象加入`ld`，然后返回

```python
ld.add_action(example - node)
ld.add_action(example - node2)
return launch_description
```

#### 调用shell命令

需要添加头文件

```python
from launch.actions import ExecuteProcess
```

使用`ExecuteProcess`调用shell命令

```python
example_cmd = ExecuteProcess(
    cmd=['some-cmd', 'some-cmd'],  # 命令，用逗号隔开
    additional_env={'EXAMPLE_PATH': path},  # 可以添加临时的环境变量
    output='screen'
)

ld.add_action(example_cmd)
```

#### 获取路径

使用`FindPackageShare`获取`package`路径

```python
from launch_ros.substitutions import FindPackageShare

# ......
package_name = "example-package"
package_name_path = FindPackageShare(package=package_name).find(package_name) 
```

或者使用`get_package_share_directory`

```python
from ament_index_python.packages import get_package_share_directory

# ......
package_name = "example-package"
package_name_path = get_package_share_directory(package_name)
```

#### 连接路径

- 使用join

```python
import os

...
# 文件
file - name = 'example-file.xxx'
# 字符串前加`f`表示可以在字符串里面使用用花括号括起来的变量和表达式，如定义好的`file-name`
file - path = os.path.join(package - path, f'example-folder/{file - name}')
# 或者使用逗号隔开
file - path = os.path.join(package - path, 'example-folder', file - name)

# 路径
dir - path = os.path.join(package - path, 'example-folder/')
```

- 使用替换

```python
from launch.substitutions import PathJoinSubstitution

...
PathJoinSubstitution([
    FindPackageShare('example-package'),
    'example-folder',
    'example-file.xxx'
])
```

#### 改变参数

一般需要用到以下两个模块

```python
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
```

`LaunchConfiguration`用于在变量中存储启动参数的值并将它们传递给所需的操作，允许我们在launch文件的任何部分获取启动参数的值。

```python
example - cfg = LaunchConfiguration('arg-name', default='true')
```

`DeclareLaunchArgument`用于定义可以从上述启动文件或控制台传递的启动参数

```python
example - arg = DeclareLaunchArgument(
    'arg-name',
    default_value='xxx',
    description='some description'
)

ld.add_action(example - arg)
```

#### 启动另一个launch文件

假设已经存在很多的单独的launch文件用于启动不同的功能，如果需要同时启动这些launch文件，可以使用`IncludeLaunchDescription`在launch文件中嵌套启动launch文件，这样可以提高复用率。

需要以下两个头文件

```python
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
```

使用`IncludeLaunchDescription`嵌套launch文件，其中同样可以使用上文所述的传递参数

```python
another - launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(launch_file_dir, 'launch-file-name.launch.py')
    ),
    launch_arguments={'arg-name': example - arg}.items()
)

ld.add_action(another - launch)
```

#### 在另一个launch文件中使用参数

我们来看一个实例：这个launch文件有一个字符串格式的路径作为参数，其中的`robot_state_publisher`需要传入`robot_description`作为参数，而这个参数需要使用`open()`，
也就是需要那个路径参数。我们自然而然会想到使用`LaunchConfiguration`，但是当你试图获取`urdf_path_cfg`的时候会发现这根本不是一个字符串。具体解决方案如下：

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.actions import OpaqueFunction


def launch_setup(context, *args, **kwargs):
    use_sim_time_cfg = LaunchConfiguration('use_sim_time')
    urdf_path_cfg = LaunchConfiguration('urdf_path')
    urdf_path = urdf_path_cfg.perform(context)

    print('\033[92m' + "robot_state_publisher: Use urdf dir: " + urdf_path + '\033[0m')

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time_cfg,
            'robot_description': robot_desc
        }]
    )

    return [
        robot_state_publisher_node,
    ]


def generate_launch_description():
    ld = LaunchDescription()

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value='robot.urdf',
        description='urdf_path'
    )

    ld.add_action(urdf_path_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
```

这种写法我个人认为极其不优雅，但是确实能解决实际问题。

强烈建议ROS加入获取参数内容的方法！！！！！

###### datetime:2023/09/13 09:36

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 3.Colcon使用进阶

基础篇中带你用gcc编译了ROS2节点。对你来说，使用CMake（GCC或Makefile）和 Python Setup打包工具依然可以完成ROS2代码的编译，那为什么还需要Colcon呢？

带着这个问题，我们来进一步的学习Colcon。

## 1.ROS生态中的构建系统和构建工具

### 1.1 构建系统与构建工具

两者的区分点在于针对的对象不同，构建系统之针对一个单独的包进行构建，而构建工具重点在于按照依赖关系依次调用构建系统完成一系列功能包的构建。

ROS中用到的构建系统：`CMake`、`ament_cmake`、`catkin`、`Python setuptools`。

ROS中用到的构建工具：`colcon`、`catkin_make`、`catkin_make_isolated`、`catkin_tools`。

很明显colcon作为构建工具，通过调用`CMake`、`Python setuptools`完成构建。

### 1.2 常见构建系统

#### 1.2.1 CMake

[CMake](https://cmake.org/) 是一个跨平台构建系统生成器。项目使用独立于平台的文件指定其生成过程。用户通过使用CMake为其平台上的本机工具生成构建系统来构建项目。

通常用法有：`cmake`、`make`、`make intsall`

#### 1.2.2 Python setuptools

`setuptools`是Python包的打包常用工具。Python
包使用文件来描述依赖项，以及如何构建和安装内容。在ROS2中，功能包可以是“普通”Python包，而在ROS1中，任何Python功能都是从CMake文件触发setup.py进行打包。

通常的用法有：`python setup.py`

#### 1.2.3 catkin

[catkin](http://wiki.ros.org/catkin)基于CMake，并提供了一组方便的函数，使编写CMake包更容易。它自动生成 CMake 配置文件以及 pkg 配置文件。它还提供了注册不同类型测试的函数。

### 1.3 常见构建工具

#### 1.3.1 catkin_make

该工具仅调用 CMake 一次，并使用 CMake
的函数在单个上下文中处理所有包。虽然这是一种有效的方法，因为所有包中的所有目标都可以并行化，但它具有明显的缺点。由于所有函数名称、目标和测试都共享一个命名空间，并且规模更大，这很容易导致冲突。

#### 1.3.2 colcon

![image-20220604133925270](imgs/image-20220604133925270.png)

[colcon](http://colcon.readthedocs.io/)是一个命令行工具，用于改进构建，测试和使用多个软件包的工作流程。它自动化了流程，处理了订购并设置了使用软件包的环境。

[colcon 文档](https://colcon.readthedocs.io/en/released/index.html)

#### 1.3.3 ament_tools

`ament_tools`由用于构建 ROS 2 包的独立 Python 3 包提供。它是为引导ROS 2项目而开发的，因此仅针对Python 3，并且可以在Linux，MacOS和Windows上运行。

`ament_tools`支持构建以下软件包：

- 带有`package.xml`文件的 ROS 2 包。
- 带有`package.xml`普通的 CMake 包。
- 没有清单文件的普通 CMake 包（从 CMake 文件中提取包名称和依赖项）。
- 带有`package.xml`文件的 Python 包。
- 没有清单文件的 Python 包（从`setup.py`文件中提取包名称和依赖项）。

## 2.Colcon构建进阶

我们平时用的最多的场景是编译功能包，所以这里重点介绍build时候的一些参数。

### 2.1 build参数

#### 2.1.0 构建指令

- `--packages-select` ，仅生成单个包（或选定的包）。
- `--packages-up-to`，构建选定的包，包括其依赖项。
- `--packages-above`，整个工作区，然后对其中一个包进行了更改。此指令将重构此包以及（递归地）依赖于此包的所有包。

#### 2.1.1.指定构建后安装的目录

可以通过 `--build-base`参数和`--install-base`，指定构建目录和安装目录。

#### 2.1.2.合并构建目录

`--merge-install`: 使用 作为所有软件包的安装前缀，而不是安装基中的软件包特定子目录。  
`--install-base`: 如果没有此选项，每个包都将提供自己的环境变量路径，从而导致非常长的环境变量值。

使用此选项时，添加到环境变量的大多数路径将相同，从而导致环境变量值更短。

#### 2.1.3.符号链接安装

启用`--symlink-install`后将不会把文拷贝到install目录，而是通过创建符号链接的方式。

#### 2.1.4.错误时继续安装

启用`--continue-on-error`，当发生错误的时候继续进行编译。

#### 2.1.5 CMake参数

`--cmake-args`，将任意参数传递给CMake。与其他选项匹配的参数必须以空格为前缀。

#### 2.1.6 控制构建线程

- `--executor EXECUTOR`
  ，用于处理所有作业的执行程序。默认值是根据所有可用执行程序扩展的优先级选择的。要查看完整列表，请调用 `colcon extensions colcon_core.executor --verbose`。

    - `sequential` [`colcon-core`]

      一次处理一个包。

    - `parallel` [`colcon-parallel-executor`]

      处理多个作业**平行**.

- `--parallel-workers NUMBER`
    - 要并行处理的最大作业数。默认值为  [os.cpu_count()](https://docs.python.org/3/library/os.html#os.cpu_count) 给出的逻辑 CPU 内核数。

#### 2.1.7 开启构建日志

使用`--log-level`可以设置日志级别，比如`--log-level  info`。

## 3.总结

有关测试相关的暂时不讲了，毕竟国内的程序员写测试的还是很少的（😉）。

--------------

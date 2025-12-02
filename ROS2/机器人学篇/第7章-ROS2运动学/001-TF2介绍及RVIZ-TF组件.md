###### datetime:2023/09/26 18:28

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)
> 
> [ros2 examples](https://github.com/ros2/examples)

# 1.tf2介绍

TF即`变换`的英文单词`TransForm`的缩写。所以`ROS`和`ROS2`中的`TF`就是指和坐标变换相关的工具。

> 在搞机器人当中，坐标变换经常用到，所以`ROS2`帮我们做了一个强大易用的TF工具

![../../_images/simple_robot.png](imgs/simple_robot.png)

## 1.发布坐标关系

我们先使用TF2的相关工具，解决上一节的手眼坐标转换问题，直观的感受一下TF2的强大。

要想让TF帮我们完成坐标变换，我们就需要告诉它坐标和坐标之间的关系。

![手眼系统](imgs/820a681b891a4719867e0c84a492d85f.png)

拿上面的手眼系统来说，我们要想获取到相机的基坐标系{B}和工具{P}之间的关系，只需要将机械臂和相机、相机和工具之间的关系告诉TF即可。

## 2.我们如何告诉TF？

可以使用tf的坐标广播工具进行广播坐标关系，广播时需要三个数据：

- 父坐标系名称（字符串）
- 子坐标系名称（字符串）
- 父子之间的变换关系（平移关系和旋转关系）

在终端中输入：

```
ros2 run tf2_ros static_transform_publisher 
```

按`enter`键，可以看到

```
A command line utility for manually sending a transform.
Usage: static_transform_publisher x y z qx qy qz qw frame_id child_frame_id 
OR 
Usage: static_transform_publisher x y z yaw pitch roll frame_id child_frame_id 
```

这是该CLI所提供的使用提示，可以看出

使用TF发布位置和姿态时，位置的描述使用的是xyz三个参数，而姿态的描述则分两种

- 第一种是四元数形式（qx qy qz qw）
- 第二种是欧拉角形式（yaw偏航角-rz pitch俯仰角-ry roll滚转角-rx），我们这里采用的是欧拉形式，绕x轴旋转采用欧拉角中的滚转角roll来描述，注意角度单位采用弧度制。

![image-20220125112120457](imgs/image-20220125112120457.png)

> 关于欧拉角和四元数的区别我们放到了姿态的多种表示章节来讲。

### 2.1 发布B到C的位姿

比如针对上面的手眼转换，广播机械臂坐标系{B}和相机坐标系{C}之间的关系。

父坐标系的名字就是B，子坐标系的名字是C，父子之间的平移关系是`0 0 3`,旋转关系是绕x轴旋转180度。

在ROS2中可以使用下面的指令发布变换，打开终端，输入下面的指令：

```
ros2 run tf2_ros static_transform_publisher 0 0 3 0 0 3.14 B C
```

如果在终端中看到下面的提示则代表发布成功

![发布位姿关系](imgs/6b0d8be89ab04fbd8d82e9cea3d882c8.png)

### 2.2 发布C到P的位姿

接着我们发布坐标系{C}到坐标系{T}的位姿

再打开一个新的终端，输入下面的命令：

```
ros2 run tf2_ros static_transform_publisher 2 1 2 0 0 0 C P
```

## 3. 监听/获取TF关系

发布也发布了，接着我们就把坐标系之间的关系打印出来，只要坐标系之间是有连接的，我们就可以使用TF求出来，使用下面的指令就可以得到机械臂基坐标系{B}和工具坐标系{P}之间的关系。

打开终端,输入命令：

```bash
ros2 run tf2_ros tf2_echo B P
```

![输出两个坐标之间关系](imgs/6c85c97787474091acda23a39a66f9d1.png)

可以看到终端中不断输出B和C之间的平移和旋转，平移采用的是xyz，基本正确，y和z的微小差异是因为我们发布变换时旋转输入的是3.14并不精确。

至于旋转部分采用的是四元数表示，关于这部分姿态的表示，之前的章节讲过。

除了使用TF获取关系外，ros2还提供很多工具来查看坐标之间的关系，大家可以在终端中输入下面的命令自行尝试。

## 4.TF常用工具

### 4.1 rqt_tf_tree

> 2022-04-26更新的，这个工具的二进制安装版本作者3月底才发布，之前没提的原因是需要源码装太麻烦了

这个工具需要我们手动安装下

```
sudo apt install ros-humble-rqt-tf-tree
```

安装完成后，再次打开rqt工具，`Plugins`->`Visualization`->`TF Tree`

接着你就可以看到这个强大的，几乎可以实时看到系统tf更新信息的工具，这个工具对于后面我们进行导航和机械臂的调试非常有帮助。

> 长的不一样没关系，这是后面补充的图。
> 
> 如果打开后没有该选项：`rm  ~/.config/ros.org/rqt_gui.ini`

![image-20220426181252686](imgs/image-20220426181252686.png)

### 4.2 tf2_monitor

查看所有的发布者和频率。

```
ros2 run tf2_ros tf2_monitor 
Gathering data on all frames for 10 seconds...

RESULTS: for all Frames

Frames:
Frame: C, published by <no authority available>, Average Delay: 3001.98, Max Delay: 3001.98
Frame: P, published by <no authority available>, Average Delay: 741.497, Max Delay: 741.497
All Broadcasters:
Node: <no authority available> 5029.14 Hz, Average Delay: 1871.74 Max Delay: 3001.98


RESULTS: for all Frames
Frames:
Frame: C, published by <no authority available>, Average Delay: 3001.98, Max Delay: 3001.98
Frame: P, published by <no authority available>, Average Delay: 741.497, Max Delay: 741.497
All Broadcasters:
Node: <no authority available> 5029.14 Hz, Average Delay: 1871.74 Max Delay: 3001.98


```

##### view_frames.py

可以生成TF的pdf，目前也有在线的实时查看工具。

```
ros2 run tf2_tools view_frames.py 
[INFO] [1636558316.667894410] [view_frames]: Listening to tf data during 5 seconds...
[INFO] [1636558321.702280144] [view_frames]: Generating graph in frames.pdf file...
[INFO] [1636558321.709904442] [view_frames]: Result:tf2_msgs.srv.FrameGraph_Response(frame_yaml="C: \n  parent: 'B'\n  broadcaster: 'default_authority'\n  rate: 10000.000\n  most_recent_transform: 0.000000\n  oldest_transform: 0.000000\n  buffer_length: 0.000\nP: \n  parent: 'C'\n  broadcaster: 'default_authority'\n  rate: 10000.000\n  most_recent_transform: 0.000000\n  oldest_transform: 0.000000\n  buffer_length: 0.000\n")
/opt/ros/humble/lib/tf2_tools/view_frames.py:75: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  data = yaml.load(result.frame_yaml)

```

![坐标树](imgs/82291e00c53c4711b6328cc4b3a9ce90.png)

除了使用命令行进行坐标关系的广播和监听，我们还可以使用代码来广播和监听，接下来就带你一起用程序来发布TF广播和获取坐标关系。

# 2.可视化坐标变换

运行上节课的示例，打开终端输入rviz2，打开rviz2，我们尝试在rviz2中直观的看到坐标之间的关系

## 1.设置默认坐标系

刚打开RVIZ2，你看到的应该是这样一个界面
![RVIZ2初始化界面](imgs/6e43fd3ebffb4436b8486d26c6b72c3b.png)
我们可以看到窗口左边的配置选项，全局选项默认选择的FixedFrame为map，这个map就是rviz2默认的坐标系的名字，动动脚趾头想一下我们并没有发布map这个坐标系，所以下面Global Status也是红色的错误。

此时我们可以手动的修改以下固定的Frame为B，让默认的坐标系设置成机械臂的基坐标系{B}。

![设置默认坐标系](imgs/f8d9586dac624df482450bb7ea57fa6f.png)

设置完成后，错误也没有了，因为此时的ROS2的TF中确确实实找到了一个叫做B的坐标系。

## 2.添加TF插件

即使没有错误，现在我们还是看不到坐标系，这是为什么呢？RVIZ2是一个插件化的软件，所以我们要添加TF相关的插件才能看到TF数据。

点击左下角的Add，在弹出的窗口中选择TF点击OK

![RVIZ2插件选择](imgs/e58d92dfda4b4819b987b7ca76785a6a.png)

之后你在RVIZ2中就可以看到下图的坐标关系

![坐标关系显示](imgs/824ad47cce374f198280e8a47e444999.png)

终于显示出来了，但是没有名字又太小了，修改下左边的选项，勾选Show Names，修改Marker Scale 为5

![修改选项](imgs/377a74a2e5d54c0589b79b0deb5d2426.png)

如果觉得视角不好，可以使用鼠标左键右键以及按下滚轮拖动修改。



--------------


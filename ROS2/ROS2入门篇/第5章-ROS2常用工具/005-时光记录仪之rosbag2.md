###### datetime:2023/09/21 10:19

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)

# 5.时光记录仪之rosbag2

本节我们来介绍ROS2中常用的一个CLI工具——rosbag2，这个工具用于记录话题的数据（就像录视频一样）。

我们就可以使用这个指令将话题数据存储为文件 ，后续我们无需启动节点，直接可以将bag文件里的话题数据发布出来。

> 这个工具在我们做一个真实机器人的时候非常有用，比如我们可以录制一段机器人发生问题的话题数据，录制完成后可以多次发布出来进行测试和实验，也可以将话题数据分享给别人用于验证算法等。

我们尝试使用bag工具来记录话题数据，并二次重放。

## 一、安装

当我们安装ROS2的时候，这个命令行工具已经为我们自动安装了，这里我们就不需要再次安装。

## 二、记录

### 2.1 常用指令

启动talker

```
ros2 run demo_nodes_cpp talker
```

#### 2.1.1 记录

`/topic-name` 为话题名字

```bash
ros2 bag record /topic-name
```

#### 2.1.2 记录多个话题的数据

```bash
ros2 bag record topic-name1  topic-name2
```

#### 2.1.3 记录所有话题

```bash
ros2 bag record -a
```

#### 2.1.4其他选项

##### `-o name` 自定义输出文件的名字

```bash
ros2 bag record -o file-name topic-name
```

##### `-s` 存储格式

目前仅支持sqllite3,其他还带拓展，后续更新再更新。

### 2.2 录制chatter

#### 2.2.1 启动talker

运行talker节点

```bash
ros2 run demo_nodes_cpp talker
```

![李四正在发布小说](imgs/005-rosbag.png)

#### 2.2.2 录制

接着使用像下面的指令就可以进行话题数据的录制了

```bash
ros2 bag record /chatter
```

如何停止录制呢？我们直接在终端中使用`Ctrl+C`指令打断录制即可

接着你会在终端中发现多处一个文件夹，名字叫做`rosbag2_xxxxxx.db3 `

打开文件夹，可以看到内容

![文件内容](imgs/7d32470a2c12477f8c90a397a9af339a.png)

这样我们就完成了录制。

## 三、查看录制出话题的信息

我们在播放一个视频前，可以通过文件信息查看视频的相关信息，比如话题记录的时间，大小，类型，数量

```bash
ros2 bag info bag-file
```

## 四、播放

### 4.1 播放话题数据

接着我们就可以重新的播放数据,使用下面的指令可以播放数据

```bash
ros2 bag play xxx.db3
```

使用ros2的topic的指令来查看数据

```bash
ros2 topic echo /chatter
```

### 4.2 播放选项

### 4.2.1 倍速播放 `-r`

`-r`选项可以修改播放速率，比如 `-r 值`，比如 `-r 10`,就是10倍速，十倍速播放话题

```bash
ros2 bag play rosbag2_2021_10_03-15_31_41_0.db3 -r 10
```

### 4.2.2 `-l` 循环播放

单曲循环就是它了

```bash
ros2 bag play rosbag2_2021_10_03-15_31_41_0.db3  -l
```

### 4.2.3 播放单个话题

```bash
ros2 bag play rosbag2_2021_10_03-15_31_41_0.db3 --topics /chatter
```

## 五、总结

相信你已经掌握了ROS2的bag工具~



--------------

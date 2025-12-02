###### datetime:2023/09/15 16:38

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)
> 
> [ros2 examples](https://github.com/ros2/examples)

# DDS进阶之Fast-DDS环境搭建

## 1.论FastDDS的三种打开方式

**FastDDS和普通ROS包一样，有二进制安装、源码编译、Docker三种安装方式。**

因为官方把二进制和Docker放到了官网。。 而且要填写个人信息才能下载。。 而且下载速度超级超级慢。。 而且不方便观摩源码。。 所以带你一起从源码进行安装。

因为DDS和ROS2相关，我们也可以使用colcon来编译，就不用cmake了(有需要cmake的自行到官网找)

## 2.源码编译安装FastDDS

下载编译DDS分为三步，第一步如果你已经安装了ROS2可以跳过。。

#### 1.安装工具和依赖库

安装工具

```
sudo apt install python3-colcon-common-extensions python3-vcstool zip openjdk-8-jdk  -y
```

安装依赖库

```
sudo apt-get install libasio-dev -y
```

#### 2.创建目录，下载仓库

```
mkdir -p fastdds_ws/src 
cd fastdds_ws && wget https://downloads.gradle-dn.com/distributions/gradle-6.4-bin.zip && unzip gradle-6.4-bin.zip 
wget http://fishros.com/tools/files/fastrtps.repos && vcs import src < fastrtps.repos
```

> [安装Fast DDS依赖项的 repos 文件时出现404：Not Found](https://fishros.org.cn/forum/topic/79/%E5%AE%89%E8%A3%85fast-dds%E4%BE%9D%E8%B5%96%E9%A1%B9%E7%9A%84-repos-%E6%96%87%E4%BB%B6%E6%97%B6%E5%87%BA%E7%8E%B0404-not-found/3?_=1650535091374)

#### 3.编译

```
colcon build
cd src/fastddsgen/ &&  gradle assemble
```

#### 最后一步:配置环境变量

xxx是你的目录前缀

```
echo 'source xxx/fastdds_ws/install/setup.bash' >> ~/.bashrc
echo 'export PATH=$PATH:xxx/fastdds_ws/gradle-6.4/bin/' >> ~/.bashrc
echo 'export DDSGEN=xxx/fastdds_ws/src/fastddsgen/scripts' >> ~/.bashrc
```

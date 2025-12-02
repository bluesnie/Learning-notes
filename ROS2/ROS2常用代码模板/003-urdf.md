###### datetime:2023/09/25 10:22

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)
> 
> [ros2 examples](https://github.com/ros2/examples)

# Gazebo常用插件

## 1.雷达

详细介绍及文章： [9.5给机器人添加激光传感器](..\..\chapt9\9.5给机器人添加激光传感器.md)

```xml

<gazebo reference="laser_link">
    <sensor name="laser_sensor" type="ray">
        <always_on>true</always_on>
        <visualize>true</visualize>
        <update_rate>5</update_rate>
        <pose>0 0 0.075 0 0 0</pose>
        <ray>
            <scan>
                <horizontal>
                    <samples>360</samples>
                    <resolution>1.000000</resolution>
                    <min_angle>0.000000</min_angle>
                    <max_angle>6.280000</max_angle>
                </horizontal>
            </scan>
            <range>
                <min>0.120000</min>
                <max>3.5</max>
                <resolution>0.015000</resolution>
            </range>
            <noise>
                <type>gaussian</type>
                <mean>0.0</mean>
                <stddev>0.01</stddev>
            </noise>
        </ray>

        <plugin name="laserscan" filename="libgazebo_ros_ray_sensor.so">
            <ros>
                <remapping>~/out:=scan</remapping>
            </ros>
            <output_type>sensor_msgs/LaserScan</output_type>
            <frame_name>laser_link</frame_name>
        </plugin>
    </sensor>
</gazebo>
```

## 2.IMU

详细介绍及文章： [9.4为FishBot添加IMU传感器.md](..\..\chapt9\9.4为FishBot添加IMU传感器.md)

```xml

<gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
        <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
            <ros>
                <namespace>/</namespace>
                <remapping>~/out:=imu</remapping>
            </ros>
            <initial_orientation_as_reference>false</initial_orientation_as_reference>
        </plugin>
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <visualize>true</visualize>
        <imu>
            <angular_velocity>
                <x>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>2e-4</stddev>
                        <bias_mean>0.0000075</bias_mean>
                        <bias_stddev>0.0000008</bias_stddev>
                    </noise>
                </x>
                <y>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>2e-4</stddev>
                        <bias_mean>0.0000075</bias_mean>
                        <bias_stddev>0.0000008</bias_stddev>
                    </noise>
                </y>
                <z>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>2e-4</stddev>
                        <bias_mean>0.0000075</bias_mean>
                        <bias_stddev>0.0000008</bias_stddev>
                    </noise>
                </z>
            </angular_velocity>
            <linear_acceleration>
                <x>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>1.7e-2</stddev>
                        <bias_mean>0.1</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </x>
                <y>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>1.7e-2</stddev>
                        <bias_mean>0.1</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </y>
                <z>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>1.7e-2</stddev>
                        <bias_mean>0.1</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </z>
            </linear_acceleration>
        </imu>
    </sensor>
</gazebo>
```

## 3.超声波

详细介绍及文章： [9.6拓展-为Fishbot添加超声波传感器.md](..\..\chapt9\9.6拓展-为Fishbot添加超声波传感器.md)

```xml

<gazebo reference="ultrasonic_sensor_link">
    <sensor type="ray" name="ultrasonic_sensor">
        <pose>0 0 0 0 0 0</pose>
        <!-- 是否可视化，gazebo里能不能看到 -->
        <visualize>true</visualize>
        <!-- 扫描速率，也就是数据更新速率 -->
        <update_rate>5</update_rate>
        <ray>
            <scan>
                <!-- 水平扫描的点数 -->
                <horizontal>
                    <samples>5</samples>
                    <resolution>1</resolution>
                    <min_angle>-0.12</min_angle>
                    <max_angle>0.12</max_angle>
                </horizontal>
                <!-- 垂直方向扫描的点数 -->
                <vertical>
                    <samples>5</samples>
                    <resolution>1</resolution>
                    <min_angle>-0.01</min_angle>
                    <max_angle>0.01</max_angle>
                </vertical>
            </scan>
            <!-- 超声波检测的范围和数据分辨率单位m -->
            <range>
                <min>0.02</min>
                <max>4</max>
                <resolution>0.01</resolution>
            </range>
            <!-- 数据噪声采用高斯噪声 -->
            <noise>
                <type>gaussian</type>
                <mean>0.0</mean>
                <stddev>0.01</stddev>
            </noise>
        </ray>
        <plugin name="ultrasonic_sensor_controller" filename="libgazebo_ros_ray_sensor.so">
            <ros>
                <!-- 重映射输出的话题名称 -->
                <remapping>~/out:=ultrasonic_sensor_1</remapping>
            </ros>
            <!-- 输出消息的类型，注意与雷达区分，这里是sensor_msgs/Range -->
            <output_type>sensor_msgs/Range</output_type>
            <!-- 射线类型，这里要写ultrasound，注意和雷达区分 -->
            <radiation_type>ultrasound</radiation_type>
            <!-- frame名称，填写link名称即可 -->
            <frame_name>ultrasonic_sensor_link</frame_name>
        </plugin>
    </sensor>
</gazebo>
```

## 4.两轮差速

详细介绍及文章： [9.3为FishBot配置两轮差速控制插件.md](..\..\chapt9\9.3为FishBot配置两轮差速控制插件.md)

```xml

<gazebo>
    <plugin name='diff_drive' filename='libgazebo_ros_diff_drive.so'>
        <ros>
            <namespace>/</namespace>
            <remapping>cmd_vel:=cmd_vel</remapping>
            <remapping>odom:=odom</remapping>
        </ros>
        <update_rate>30</update_rate>
        <!-- wheels -->
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <!-- kinematics -->
        <wheel_separation>0.2</wheel_separation>
        <wheel_diameter>0.065</wheel_diameter>
        <!-- limits -->
        <max_wheel_torque>20</max_wheel_torque>
        <max_wheel_acceleration>1.0</max_wheel_acceleration>
        <!-- output -->
        <publish_odom>true</publish_odom>
        <publish_odom_tf>true</publish_odom_tf>
        <publish_wheel_tf>true</publish_wheel_tf>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>base_footprint</robot_base_frame>
    </plugin>
</gazebo>
```

## 5.JointStatePublisher

待补充

## 6.单目相机

```xml

<gazebo reference="camera_link">
    <sensor type="camera" name="camera">
        <update_rate>30.0</update_rate>
        <camera name="head">
            <horizontal_fov>1.3962634</horizontal_fov>
            <image>
                <width>800</width>
                <height>800</height>
                <format>R8G8B8</format>
            </image>
            <clip>
                <near>0.02</near>
                <far>300</far>
            </clip>
            <noise>
                <type>gaussian</type>
                <mean>0.0</mean>
                <stddev>0.007</stddev>
            </noise>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
            <alwaysOn>true</alwaysOn>
            <updateRate>0.0</updateRate>
            <cameraName>/camera</cameraName>
            <imageTopicName>image_raw</imageTopicName>
            <cameraInfoTopicName>camera_info</cameraInfoTopicName>
            <frameName>camera_link</frameName>
            <hackBaseline>0.07</hackBaseline>
        </plugin>
    </sensor>
    <material>Gazebo/Blue</material>
</gazebo>
```

## 7.深度相机

待补充

# ROS2中常用的Xacro模板

URDF默认格式是纯文本的，我们并不能在其中加入计算公式和定义，用URDF定义一个机器人模型会导致整个文件非常冗长，使用Xacro工具可以解决这个问题。

Xacro是urdf的定义和生成工具，你按照Xacro提供的方式定义可以复用的模型描述块，之后就可以直接调用这些描述，最后使用xacro指令生成最终的urdf模型了。

## 1.添加模板

这里提供了常用的xacro描述定义的代码块，你可以直接引入的你的工程里进行使用。

在你的功能包里新建`xacro_template.xacro`文件，复制粘贴下面的内容到其中。

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" xmlns:fishros="http://fishros.com">
    <xacro:macro name="box_inertia" params="m w h d">
        <inertial>
            <origin xyz="0 0 0" rpy="${pi/2} 0 ${pi/2}"/>
            <mass value="${m}"/>
            <inertia ixx="${(m/12) * (h*h + d*d)}" ixy="0.0" ixz="0.0" iyy="${(m/12) * (w*w + d*d)}" iyz="0.0"
                     izz="${(m/12) * (w*w + h*h)}"/>
        </inertial>
    </xacro:macro>

    <xacro:macro name="cylinder_inertia" params="m r h">
        <inertial>
            <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
            <mass value="${m}"/>
            <inertia ixx="${(m/12) * (3*r*r + h*h)}" ixy="0" ixz="0" iyy="${(m/12) * (3*r*r + h*h)}" iyz="0"
                     izz="${(m/2) * (r*r)}"/>
        </inertial>
    </xacro:macro>

    <xacro:macro name="sphere_inertia" params="m r">
        <inertial>
            <mass value="${m}"/>
            <inertia ixx="${(2/5) * m * (r*r)}" ixy="0.0" ixz="0.0" iyy="${(2/5) * m * (r*r)}" iyz="0.0"
                     izz="${(2/5) * m * (r*r)}"/>
        </inertial>
    </xacro:macro>

    <xacro:macro name="sphere_visual" params="r origin_r origin_p origin_y">
        <visual>
            <origin xyz="0 0 0" rpy="${origin_r} ${origin_p} ${origin_y}"/>
            <geometry>
                <sphere radius="${r}"/>
            </geometry>
            <material name="blue">
                <color rgba="0.0 0.0 0.8 1.0"/>
            </material>
        </visual>
    </xacro:macro>

    <xacro:macro name="sphere_collision" params="r origin_r origin_p origin_y">
        <collision>
            <origin xyz="0 0 0" rpy="${origin_r} ${origin_p} ${origin_y}"/>
            <geometry>
                <sphere radius="${r}"/>
            </geometry>
            <material name="green">
                <color rgba="0.0 0.8 0.0 1.0"/>
            </material>
        </collision>
    </xacro:macro>


    <xacro:macro name="box_visual" params="w d h origin_r origin_p origin_y">
        <visual>
            <origin xyz="0 0 0" rpy="${origin_r} ${origin_p} ${origin_y}"/>
            <geometry>
                <box size="${w} ${d} ${h}"/>
            </geometry>
            <material name="blue">
                <color rgba="0.0 0.0 0.8 1.0"/>
            </material>
        </visual>
    </xacro:macro>

    <xacro:macro name="box_collision" params="w d h origin_r origin_p origin_y">
        <collision>
            <origin xyz="0 0 0" rpy="${origin_r} ${origin_p} ${origin_y}"/>
            <geometry>
                <box size="${w} ${d} ${h}"/>
            </geometry>
            <material name="green">
                <color rgba="0.0 0.8 0.0 1.0"/>
            </material>
        </collision>
    </xacro:macro>


    <xacro:macro name="cylinder_visual" params="r h origin_r origin_p origin_y">
        <visual>
            <origin xyz="0 0 0" rpy="${origin_r} ${origin_p} ${origin_y}"/>
            <geometry>
                <cylinder length="${h}" radius="${r}"/>
            </geometry>
            <material name="blue">
                <color rgba="0.0 0.0 0.8 1.0"/>
            </material>
        </visual>
    </xacro:macro>


    <xacro:macro name="cylinder_collision" params="r h origin_r origin_p origin_y">
        <collision>
            <origin xyz="0 0 0" rpy="${origin_r} ${origin_p} ${origin_y}"/>
            <geometry>
                <cylinder length="${h}" radius="${r}"/>
            </geometry>
            <material name="blue">
                <color rgba="0.0 0.0 0.8 1.0"/>
            </material>
        </collision>
    </xacro:macro>


</robot>
```

## 2.使用模板生成URDF

接着你可以新建你的机器人模型描述文件，比如`fishbot.urdf.xacro`,之后你就可以在你的描述文件中调用提供的模板，快速的定义机器人。

比如创建一个正方体的base_link,并导入惯性矩阵。

```xml
<?xml version="1.0"?>
<robot name="fishbot" xmlns:xacro="http://ros.org/wiki/xacro">
    <xacro:include filename="xacro_template.xacro"/>

    <link name="base_link">
        <xacro:box_visual w="0.809" d="0.5" h="0.1" origin_r="0" origin_p="0" origin_y="0"/>
        <xacro:box_collision w="0.809" d="0.5" h="0.1" origin_r="0" origin_p="0" origin_y="0"/>
        <xacro:box_inertia m="1.0" w="0.809" d="0.5" h="0.1"/>
    </link>

</robot>
```

上面w,d,h,代表长宽高，m代表质量。` <xacro:include filename="xacro_template.xacro" />`用于引入提供的模板。

接着我们就可以通过xacro指令将其变成正常的urdf，打开终端，进入fishbot.urdf.xacro同级目录，输入指令`xacro fishbot.urdf.xacro -o fishbot.urdf`
,即可生成fishbot.urdf，正常生成后的内容如下。

```xml
<?xml version="1.0" ?>
<!-- =================================================================================== -->
<!-- |    This document was autogenerated by xacro from fishbot.urdf.xacro             | -->
<!-- |    EDITING THIS FILE BY HAND IS NOT RECOMMENDED                                 | -->
<!-- =================================================================================== -->
<robot name="fishbot" xmlns:fishros="http://fishros.com">
    <link name="base_link">
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <box size="0.809 0.5 0.1"/>
            </geometry>
            <material name="blue">
                <color rgba="0.0 0.0 0.8 1.0"/>
            </material>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <box size="0.809 0.5 0.1"/>
            </geometry>
            <material name="green">
                <color rgba="0.0 0.8 0.0 1.0"/>
            </material>
        </collision>
        <inertial>
            <origin rpy="1.5707963267948966 0 1.5707963267948966" xyz="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.021666666666666667" ixy="0.0" ixz="0.0" iyy="0.07537341666666666" iyz="0.0"
                     izz="0.055373416666666675"/>
        </inertial>
    </link>
</robot>
```

这就是xacro的神奇之处，将短短的三行定义根据规则生成长长的URDF，关于xacro的详细使用可以参考 http://ros.org/wiki/xacro 。
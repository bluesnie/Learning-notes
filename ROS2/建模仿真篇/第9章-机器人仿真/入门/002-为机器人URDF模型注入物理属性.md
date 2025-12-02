###### datetime:2023/09/26 18:28

###### author:nzb

> 该项目来源于[大佬的动手学ROS2](https://fishros.com/d2lros2)
> 
> [ros2 examples](https://github.com/ros2/examples)

# 9.2.为机器人URDF模型注入物理属性

上节我们知道，机器人仿真就是用软件来模拟硬件的特性，那么我们必须要告诉仿真平台机器人各个关节的物理属性，比如：

- 有多重，
- 有多大的惯性
- 重心在哪
- 碰撞边界在哪
- 关节的上下界限
- 其他的一些必要信息等等

所以这节课就带你将物理信息写入到urdf中，让机器人在gazebo中显示出来。

## 1.需要哪些物理信息？

一般来说有`碰撞`和`内参`两个就够了，但是因为之前的偷懒，还要加一个摩擦力配置。

碰撞描述是物体的用于碰撞检测的包围形状。内参用于描述物体的质量，惯性矩阵。link的摩擦力。

### 1.1 碰撞检测

在机器人仿真中，我们要对物体之前是否接触，是否发生碰撞做检测，常用的检测方法比如包围盒，判断两个物体的包围盒是否相交来快速判断物体是否发生碰撞。

在URDF中，我们可以可以在link标签下添加collison子标签来对物体的形状进行描述。

collision可以包含的子标签如下：

- origin，表示碰撞体的中心位姿
- geometry，用于表示用于碰撞检测的几何形状
- `material`，可选的，描述碰撞几何体的材料(这个设置可以在gazebo仿真时通过view选项看到碰撞包围体的形状)

一个完整的collision标签实例如下：

```xml

<collision>
    <origin xyz="0 0 0.0" rpy="0 0 0"/>
    <geometry>
        <cylinder length="0.12" radius="0.10"/>
    </geometry>
    <material name="blue">
        <color rgba="0.1 0.1 1.0 0.5"/>
    </material>
</collision>
```

### 1.2 旋转惯量

旋转惯量矩阵是用于描述物体的惯性的，在做动力学仿真的时候，这些参数尤为重要。

在URDF中我们可以通过在link下添加inertial子标签，为link添加惯性参数的描述。

intertial标签包含的子标签如下：

- mass，描述link的质量
- inertia，描述link的旋转惯量（该标签有六个属性值ixx\ixy\ixz\iyy\iyz\izz）

一个完整的inertial标签示例如下：

```
   <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0122666" ixy="0" ixz="0" iyy="0.0122666" iyz="0" izz="0.02"/>
    </inertial>
```

关于intertial的属性设置，不是随意设置的，常见的几何体我们可以通过公式进行计算。计算方法可以看的这篇文章-[URDF仿真惯性参数不知道怎么配？快收藏，常见几何物体URDF分享](https://mp.weixin.qq.com/s/3L8Lilesy2W_WY5qup0gmA)。

比如我们上一章节的fishbot的轮子和车体，都是实心圆柱，可以采用下面的公式进行计算：

> 注意：这个矩阵是一个对称矩阵，所以只需要通过其上三角即可描述完整描述这个矩阵，所以在URDF中只需要填写六个数字即可。
> 实心圆柱体的惯性矩阵:半径为r，高度为h，质量为m 的实心圆柱体
> 
> 形状：![实心圆柱体](imgs/453a07bf69814351a7c673deddf78087.png)
> 
> 矩阵：![矩阵](imgs/9b7f85f6d4314130ae6abdce6beeae8f.png)

### 1.3 摩擦力和刚性系数

在Fishbot的URDF中，前面的支撑轮主要起支撑作用，因为我们将其使用fixed标签固定到了base_link上，所以它无法转动。

哪该怎么办呢？教你一个取巧的办法，我们可以要把这个轮子的摩擦力设置为0，让它直接在地上滑动即可，

如何设置呢？6行代码放到URDF中：

```
  <gazebo reference="caster_link">
    <mu1 value="0.0"/>
    <mu2 value="0.0"/>
    <kp value="1000000.0" />
    <kd value="10.0" />
  </gazebo>
```

其中mu1,mu2代表摩擦力，kp,kd代表刚性系数。

## 2.为FishBot添加物理惯性

利用上面的方法公式，为我们的fishbot个个link添加好物理属性，完成后的base_link如下：

```xml
  <!-- base link -->
<link name="base_link">
    <visual>
        <origin xyz="0 0 0.0" rpy="0 0 0"/>
        <geometry>
            <cylinder length="0.12" radius="0.10"/>
        </geometry>
        <material name="blue">
            <color rgba="0.1 0.1 1.0 0.5"/>
        </material>
    </visual>
    <collision>
        <origin xyz="0 0 0.0" rpy="0 0 0"/>
        <geometry>
            <cylinder length="0.12" radius="0.10"/>
        </geometry>
        <material name="blue">
            <color rgba="0.1 0.1 1.0 0.5"/>
        </material>
    </collision>
    <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.0122666" ixy="0" ixz="0" iyy="0.0122666" iyz="0" izz="0.02"/>
    </inertial>
</link>
```

完全添加好的机器人URDF模型

```xml
<?xml version="1.0"?>
<robot name="fishbot">


    <!-- Robot Footprint -->
    <link name="base_footprint"/>

    <joint name="base_joint" type="fixed">
        <parent link="base_footprint"/>
        <child link="base_link"/>
        <origin xyz="0.0 0.0 0.076" rpy="0 0 0"/>
    </joint>


    <!-- base link -->
    <link name="base_link">
        <visual>
            <origin xyz="0 0 0.0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.12" radius="0.10"/>
            </geometry>
            <material name="blue">
                <color rgba="0.1 0.1 1.0 0.5"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0.0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.12" radius="0.10"/>
            </geometry>
            <material name="blue">
                <color rgba="0.1 0.1 1.0 0.5"/>
            </material>
        </collision>
        <inertial>
            <mass value="0.2"/>
            <inertia ixx="0.0122666" ixy="0" ixz="0" iyy="0.0122666" iyz="0" izz="0.02"/>
        </inertial>
    </link>

    <!-- laser link -->
    <link name="laser_link">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.02" radius="0.02"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.02" radius="0.02"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.000190416666667" ixy="0" ixz="0" iyy="0.0001904" iyz="0" izz="0.00036"/>
        </inertial>
    </link>

    <!-- laser joint -->
    <joint name="laser_joint" type="fixed">
        <parent link="base_link"/>
        <child link="laser_link"/>
        <origin xyz="0 0 0.075"/>
    </joint>

    <link name="imu_link">
        <visual>
            <origin xyz="0 0 0.0" rpy="0 0 0"/>
            <geometry>
                <box size="0.02 0.02 0.02"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0.0" rpy="0 0 0"/>
            <geometry>
                <box size="0.02 0.02 0.02"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.000190416666667" ixy="0" ixz="0" iyy="0.0001904" iyz="0" izz="0.00036"/>
        </inertial>
    </link>

    <!-- imu joint -->
    <joint name="imu_joint" type="fixed">
        <parent link="base_link"/>
        <child link="imu_link"/>
        <origin xyz="0 0 0.02"/>
    </joint>


    <link name="left_wheel_link">
        <visual>
            <origin xyz="0 0 0" rpy="1.57079 0 0"/>
            <geometry>
                <cylinder length="0.04" radius="0.032"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="1.57079 0 0"/>
            <geometry>
                <cylinder length="0.04" radius="0.032"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </collision>
        <inertial>
            <mass value="0.2"/>
            <inertia ixx="0.000190416666667" ixy="0" ixz="0" iyy="0.0001904" iyz="0" izz="0.00036"/>
        </inertial>
    </link>

    <link name="right_wheel_link">
        <visual>
            <origin xyz="0 0 0" rpy="1.57079 0 0"/>
            <geometry>
                <cylinder length="0.04" radius="0.032"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="1.57079 0 0"/>
            <geometry>
                <cylinder length="0.04" radius="0.032"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </collision>
        <inertial>
            <mass value="0.2"/>
            <inertia ixx="0.000190416666667" ixy="0" ixz="0" iyy="0.0001904" iyz="0" izz="0.00036"/>
        </inertial>
    </link>

    <joint name="left_wheel_joint" type="continuous">
        <parent link="base_link"/>
        <child link="left_wheel_link"/>
        <origin xyz="-0.02 0.10 -0.06"/>
        <axis xyz="0 1 0"/>
    </joint>

    <joint name="right_wheel_joint" type="continuous">
        <parent link="base_link"/>
        <child link="right_wheel_link"/>
        <origin xyz="-0.02 -0.10 -0.06"/>
        <axis xyz="0 1 0"/>
    </joint>

    <link name="caster_link">
        <visual>
            <origin xyz="0 0 0" rpy="1.57079 0 0"/>
            <geometry>
                <sphere radius="0.016"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="1.57079 0 0"/>
            <geometry>
                <sphere radius="0.016"/>
            </geometry>
            <material name="black">
                <color rgba="0.0 0.0 0.0 0.5"/>
            </material>
        </collision>
        <inertial>
            <mass value="0.02"/>
            <inertia ixx="0.000190416666667" ixy="0" ixz="0" iyy="0.0001904" iyz="0" izz="0.00036"/>
        </inertial>
    </link>

    <joint name="caster_joint" type="fixed">
        <parent link="base_link"/>
        <child link="caster_link"/>
        <origin xyz="0.06 0.0 -0.076"/>
        <axis xyz="0 1 0"/>
    </joint>


    <gazebo reference="caster_link">
        <material>Gazebo/Black</material>
    </gazebo>

    <gazebo reference="caster_link">
        <mu1 value="0.0"/>
        <mu2 value="0.0"/>
        <kp value="1000000.0"/>
        <kd value="10.0"/>
        <!-- <fdir1 value="0 0 1"/> -->
    </gazebo>


    <gazebo>
        <plugin name='diff_drive' filename='libgazebo_ros_diff_drive.so'>
            <ros>
                <namespace>/</namespace>
                <remapping>cmd_vel:=cmd_vel</remapping>
                <remapping>odom:=odom</remapping>
            </ros>
            <update_rate>30</update_rate>
            <!-- wheels -->
            <!-- <left_joint>left_wheel_joint</left_joint> -->
            <!-- <right_joint>right_wheel_joint</right_joint> -->
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
            <publish_wheel_tf>false</publish_wheel_tf>
            <odometry_frame>odom</odometry_frame>
            <robot_base_frame>base_footprint</robot_base_frame>
        </plugin>


        <plugin name="fishbot_joint_state" filename="libgazebo_ros_joint_state_publisher.so">
            <ros>
                <remapping>~/out:=joint_states</remapping>
            </ros>
            <update_rate>30</update_rate>
            <joint_name>right_wheel_joint</joint_name>
            <joint_name>left_wheel_joint</joint_name>
        </plugin>
    </gazebo>

    <gazebo reference="laser_link">
        <material>Gazebo/Black</material>
    </gazebo>

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
                    <!-- <namespace>/tb3</namespace> -->
                    <remapping>~/out:=scan</remapping>
                </ros>
                <output_type>sensor_msgs/LaserScan</output_type>
                <frame_name>laser_link</frame_name>
            </plugin>
        </sensor>
    </gazebo>

</robot>
```

可以将配置好的模型下载到`src/c/urdf`文件夹下，等下我们要用gazebo将该模型显示出来。
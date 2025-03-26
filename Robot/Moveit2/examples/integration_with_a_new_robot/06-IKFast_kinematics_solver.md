###### datetime:2025/03/05 11:27

###### author:nzb

# [IKFast Kinematics Solver](https://moveit.picknik.ai/main/doc/examples/ikfast/ikfast_tutorial.html)


## 什么是 IKFast？

机器人运动学编译器 `IKFast` 是罗森戴安科夫公司的 `OpenRAVE` 运动规划软件中提供的一个功能强大的逆运动学求解器。`IKFast` 可自动分析任何复杂的运动学链，找出可用于分析求解的常见模式，并生成 `C++` 代码来找到这些模式。因此，`IKFast` 可提供极其稳定的解决方案，在最新的处理器上只需几微秒即可找到。

## MoveIt IKFast

`MoveIt` 提供了使用 `OpenRAVE` 生成的 `cpp` 文件为 `MoveIt` 生成 `IKFast` 运动学插件的工具。本教程将指导您如何设置机器人以利用 `IKFast` 的强大功能。`MoveIt IKFast` 在 `ROS Melodic` 上使用 `6DOF` 和 `7DOF` 机械臂操纵器进行了测试。虽然理论上可行，**但 `MoveIt IKFast` 目前并不支持大于 7 自由度的机械臂。**

## 开始

您应该有一个通过 "设置助手 "为机器人创建的 `MoveIt` 配置包。

`OpenRAVE` 是一个与 `MoveIt` 本身一样复杂的规划框架，安装起来非常棘手，尤其是因为其公开文档已不再维护。幸运的是，`personalrobotics` 提供了一个基于 `Ubuntu 14.04` 的 [`docker` 镜像](https://hub.docker.com/r/personalrobotics/ros-openrave)，其中安装了 `OpenRAVE 0.9.0` 和 `ROS Indigo`，可用于一次性生成求解器代码。

本教程中提供的命令会自动下载并启动该 `Docker` 镜像，因此暂时不需要额外的步骤。

## 创建 IKFast MoveIt 插件

为便于复制和粘贴，我们建议将机器人名称定义为环境变量：

```bash
export ROBOT_NAME=my_robot
```

`OpenRAVE` 使用 `Collada` 代替 `URDF` 来描述机器人。为了自动将机器人的 `URDF` 转换为 `Collada`，您需要提供 `.urdf` 文件。如果您的 `.urdf` 文件是由 `xacro` 文件生成的，您可以使用以下命令生成 `URDF`：

```bash
ros2 run xacro xacro -o $MYROBOT_NAME.urdf $MYROBOT_NAME.urdf.xacro
```

### 选择 IK 类型

您需要选择要求解的 `IK` 类型。更多信息请参见[本页](http://openrave.org/docs/latest_stable/openravepy/ikfast/#ik-types)。最常见的 `IK` 类型是 **transform6d** 。

### 选择规划组

如果您的机器人有多个手臂或 "规划组"，而您希望为其生成 `IKFast` 求解器，则需要为每个规划组重复以下步骤。以下说明将假定您选择了一个`<planning_group_name>`。此外，您还需要知道要求解的链条的基础链节和执行器链节的名称。

### 生成 IKFast Moveit 插件

```bash
ros2 run moveit_kinematics auto_create_ikfast_moveit_plugin.sh --iktype Transform6D $MYROBOT_NAME.urdf <planning_group_name> <base_link> <eef_link>
```

这一过程的速度和成功与否取决于机器人的复杂程度。一个典型的 `6 DOF` 机械手，在底部或手腕处有 `3` 个交叉轴，只需几分钟即可生成求解器代码。有关创建过程的详细说明和其他调整，请参阅创建过程的调整。

上述命令会在当前文件夹中创建一个名为 `$MYROBOT_NAME_<planning_group_name>_ikfast_plugin` 的新 `ROS` 软件包。因此，您需要重建工作区，以便检测到新软件包：

```bash
colcon build
```

## 使用

`IKFast` 插件可以直接替代默认的 `KDL IK` 求解器，但性能会大大提高。生成器脚本应自动编辑 `MoveIt` 配置文件，但在某些情况下可能会失败。在这种情况下，您可以使用机器人 `kinematics.yaml` 文件中的 `kinematics_solver` 参数在 KDL 和 `IKFast` 解算器之间进行切换：

```yaml
orca_left_arm:
  # kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver: orca_orca_left_arm/IKFastKinematicsPlugin   # 注意不是：orca_orca_left_arm_ikfast_plugin/IKFastKinematicsPlugin
  kinematics_solver_search_resolution: 0.0050000000000000001
  kinematics_solver_timeout: 0.0050000000000000001
orca_right_arm:
  # kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver: orca_orca_right_arm/IKFastKinematicsPlugin
  kinematics_solver_search_resolution: 0.0050000000000000001
  kinematics_solver_timeout: 0.0050000000000000001
```

### 测试

使用 `MoveIt RViz` 运动规划插件，并使用交互式标记查看是否找到正确的 `IK` 解决方案。

## 更新插件

如果将来 `MoveIt` 或 `IKFast` 发生任何变化，您可能需要使用我们的脚本重新生成该插件。为方便起见，我们在您的 `IKFast MoveIt` 软件包根目录下自动创建了一个名为 `update_ikfast_plugin.sh` 的 `bash` 脚本。该脚本将从 `OpenRAVE` 生成的 `.cpp` 求解器文件中重新生成插件。

## 调整创建过程

`IKFast MoveIt` 插件的创建过程包括几个步骤，由创建脚本逐一执行：

- 下载由 `personalrobotics` 提供的 `docker` 镜像

- 将 `ROS URDF` 文件转换为 `OpenRAVE` 所需的 `Collada` 文件：`ros2 run collada_urdf urdf_to_collada $MYROBOT_NAME.urdf $MYROBOT_NAME.dae`

    在将 `URDF` 文件转换为 `Collada` 时，有时会出现浮点问题，导致 `OpenRAVE` 无法找到 `IK` 解决方案。使用实用脚本，可以轻松地将 `.dae` 文件中的所有数字舍入到小数点后 `n` 位。根据经验，我们建议小数点后保留 5 位，但如果 `OpenRave ikfast` 生成器寻找解决方案的时间过长（比如超过一小时），降低精度应该会有所帮助。例如

    ```bash
    ros2 run moveit_kinematics round_collada_numbers.py $MYROBOT_NAME.dae $MYROBOT_NAME.rounded.dae 5
    ```

- 运行 `OpenRAVE IKFast` 工具生成 `C++` 求解器代码
- 创建封装生成求解器的 `MoveIt IKFast` 插件包

`auto_create_ikfast_moveit_plugin.sh` 脚本会评估输入文件的扩展名，以决定运行哪些步骤。要从任何中间步骤重新运行脚本（例如，在调整了 `.dae` 文件的精确度后），只需提供前一步骤的相应输出作为输入（`.dae` 或 `.cpp`），而不是初始的 `.urdf` 文件。

## 问题

### 问题1：生成插件的时候报错

```text
Running /opt/ros/humble/lib/moveit_kinematics/create_ikfast_moveit_plugin.py "urdfhead1.5.SLDASM" "orca_left_arm" "urdfhead1.5.SLDASM_orca_left_arm_ikfast_plugin" "base_link" "lhand_tcp" "/tmp/ikfast.gqh3Pg/.openrave/kinematics.1bdabd77f9396ebcdbf6b274d52c5ea6/ikfast0x10000049.Transform6D.0_1_3_4_5_6_f2.cpp"
/usr/bin/env: ‘python’: No such file or directory
[ros2run]: Process exited with failure 127
```
- 解决

看看是不是本机缺失 `python` ,如果是 `python3` ,链接一下`ln /usr/bin/python3 /usr/bin/python`



### 问题2： colcon build 报错

```text
CMake Error at CMakeLists.txt:21 (add_library):
  Target "urdfhead_orca_left_arm_moveit_ikfast_plugin" links to target
  "eigen_stl_containers::eigen_stl_containers" but the target was not found.
  Perhaps a find_package() call is missing for an IMPORTED target, or an
  ALIAS target is missing?
```
- 解决

```bash
sudo apt install ros-humble-eigen-stl-containers
```

###### datetime:2025/04/25 16:39

###### author:nzb

# [**`move_group` 节点**](https://moveit.picknik.ai/main/doc/concepts/move_group.html)

下图展示了MoveIt核心节点`move_group`的系统架构。该节点作为集成器，将所有组件整合在一起，为用户提供一组`ROS Action`和`Service`接口。  

![](../imgs/move_group.png)

## **用户接口（User Interface）**  
用户可通过两种方式访问`move_group`的功能：  
1. **C++接口**：使用`move_group_interface`包提供的C++接口。  
2. **图形界面**：通过Rviz的**Motion Planning插件**（ROS可视化工具）。  


## **配置（Configuration）**  
`move_group`作为ROS节点，通过ROS参数服务器获取三类信息：  
1. **URDF**：从`robot_description`参数加载机器人URDF模型。  
1. **SRDF**：从`robot_description_semantic`参数加载语义描述文件（SRDF），该文件通常由MoveIt Setup Assistant生成。  
2. **MoveIt配置**：加载关节限位、运动学、运动规划等配置文件（由`MoveIt Setup Assistant`自动生成，存储于机器人配置包的`config`目录）。  


## **机器人接口（Robot Interface）**  
`move_group`通过ROS话题和Action与机器人交互：  
- **状态获取**：监听`/joint_states`话题获取关节位置（支持多发布者的部分状态信息，如机械臂与移动底盘分开发布）。  
- **坐标变换**：通过ROS TF库监听全局位姿信息（如导航栈发布的`map`到`base_link`变换）。  
- **控制接口**：通过`FollowJointTrajectoryAction`与机器人控制器通信（需机器人端部署Action服务端）。  

> **注**：  
> - `move_group`不发布关节状态或TF信息，需依赖`robot_state_publisher`节点。  
> - 控制器接口需用户自行实现Action服务端。  


## **规划场景（Planning Scene）**  
`move_group`通过**Planning Scene Monitor**维护规划场景，包含：  
- 环境模型  
- 机器人当前状态（包括刚性附着物体）。  
详细架构见后续“规划场景”章节。  

## **可扩展能力（Extensible Capabilities）**  
`move_group`采用插件化设计，支持以下功能的灵活扩展：  
- 抓取放置（Pick and Place）  
- 运动学求解（Kinematics）  
- 运动规划（Motion Planning）  

插件通过ROS参数和`pluginlib`库配置，普通用户无需手动调整（MoveIt Setup Assistant生成的启动文件已自动配置）。  

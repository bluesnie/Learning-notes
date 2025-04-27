###### datetime:2025/04/23 11:47

###### author:nzb

# [**运动学 (Kinematics)**](https://moveit.picknik.ai/main/doc/concepts/kinematics.html)

## **运动学插件**

`MoveIt`采用插件化架构，特别支持用户自定义逆运动学算法。正运动学和雅可比矩阵计算已集成在`RobotState`类中。`MoveIt`默认的逆运动学插件使用**基于KDL数值雅可比求解器**的算法，该插件由`MoveIt Setup Assistant`自动配置。

### **碰撞检测 (Collision Checking)**
`MoveIt`中的碰撞检测通过`Planning Scene`中的`CollisionWorld`对象实现。其设计使用户无需关注底层实现细节，主要依赖**FCL库**（`MoveIt`的核心碰撞检测库）完成。

## **碰撞对象 (Collision Objects)**
`MoveIt`支持以下类型的碰撞检测：
- **网格模型(Meshes)**：可用`.stl`（标准三角语言）或`.dae`（数字资产交换）格式描述机器人连杆等物体。  
- **基本几何体(Primitive Shapes)**：如立方体、圆柱体、圆锥体、球体和平面。  
- **Octomap**：可直接用于碰撞检测的八叉树地图。  

## **允许碰撞矩阵 (Allowed Collision Matrix, ACM)**  

碰撞检测是运动规划中计算量最大的环节（约占90%资源消耗）。ACM通过二进制值定义物体对（机器人本体或环境物体）之间是否需要检测碰撞：  
- 若某对物体的ACM值为`true`，则跳过二者碰撞检测。  
- 适用场景：
  - 两物体距离极远，不可能发生碰撞。  
  - 两物体默认处于接触状态（需主动禁用碰撞检测）。  

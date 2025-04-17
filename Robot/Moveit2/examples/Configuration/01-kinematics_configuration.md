###### datetime:2025/03/27 11:48

###### author:nzb

# 运动学配置

## `kinematics.yaml` 文件

由 `MoveIt` 配置助手生成的 `kinematics.yaml` 文件是 `MoveIt` 的主要运动学配置文件。您可以在 `Panda` 机器人的 `GitHub` 项目 `panda_moveit_config` 中查看完整示例：

```yaml
panda_arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
  kinematics_solver_attempts: 3
```

### 参数

- 可用参数包括：

- **`kinematics_solver`**：运动学求解器插件的名称。需与插件描述文件中定义的名称一致（例如 `[example_kinematics/ExampleKinematicsPlugin]`）。
- **`kinematics_solver_search_resolution`**：指定求解器在冗余空间中搜索逆运动学的分辨率（例如，对于 7 自由度机械臂，指定冗余关节的搜索步长）。
- **`kinematics_solver_timeout`**：单次逆运动学求解的超时时间（单位：秒）。典型的求解过程包括从种子状态随机重启和求解周期（此超时适用于单个周期）。
- **`kinematics_solver_attempts`**：求解器的随机重启次数。每次重启后的求解周期将使用上述超时时间。通常建议设置较短的超时时间以快速失败。

### KDL 运动学插件

`KDL` 运动学插件封装了 `Orocos KDL` 包提供的数值逆运动学求解器。

- 这是 `MoveIt` 默认使用的运动学插件。
- 遵守 `URDF` 中指定的关节限制（若 `URDF` 定义了安全限制，则优先使用）。
- 当前仅支持串联链式结构。

### LMA 运动学插件

`LMA（Levenberg-Marquardt）`运动学插件同样基于 `Orocos KDL` 包的数值求解器。

- 遵守 `URDF` 中的关节限制。
- 当前仅支持串联链式结构。
- 用法示例：`kinematics_solver: lma_kinematics_plugin/LMAKinematicsPlugin`

### 缓存式 IK 插件

缓存式 IK 插件通过持久化缓存 IK 解来加速其他 IK 求解器。调用 IK 求解器时，会从缓存中选取相似状态作为种子。若失败，则使用用户指定的种子状态重新求解。新的 IK 解若与缓存差异显著，则被加入缓存。缓存会定期保存至磁盘。

**启用缓存式 IK 插件**：需修改 `kinematics.yaml`，例如将：

```yaml
manipulator:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
```

改为：

```yaml
manipulator:
  kinematics_solver: cached_ik_kinematics_plugin/CachedKDLKinematicsPlugin
  # 可选缓存参数：
  max_cache_size: 10000    # 最大缓存条目数
  min_pose_distance: 1     # 末端位姿最小距离阈值
  min_joint_config_distance: 4  # 关节配置最小距离阈值
```
---

缓存大小可通过绝对上限（`max_cache_size`）或末端位姿距离（`min_pose_distance`）、关节状态距离（`min_joint_config_distance`）控制。缓存文件默认保存至 `$(HOME)/.ros` 的子目录中。支持的 `kinematics_solver` 值包括：

- **`cached_ik_kinematics_plugin/CachedKDLKinematicsPlugin`**：默认 KDL 求解器的封装。
- **`cached_ik_kinematics_plugin/CachedSrvKinematicsPlugin`**：通过 ROS 服务调用外部 IK 求解器的封装。
- **`cached_ik_kinematics_plugin/CachedTRACKinematicsPlugin`**：TRAC IK 求解器的封装（需编译时检测到 TRAC IK 插件）。
- **`cached_ik_kinematics_plugin/CachedUR5KinematicsPlugin`**：UR5 机械臂解析式 IK 求解器的封装（仅为示例，缓存会增加额外开销）。

详见《缓存式 IK 使用说明》。

## 仅位置逆运动学（Position Only IK）

若使用 KDL 插件，可通过在 `kinematics.yaml` 中添加以下行启用仅位置逆运动学（针对特定运动组）：

```yaml
position_only_ik: True
```

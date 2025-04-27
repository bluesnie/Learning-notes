###### datetime:2025/04/25 16:39

###### author:nzb

# [轨迹处理](https://moveit.picknik.ai/main/doc/concepts/trajectory_processing.html)

## 时间参数化

运动规划器通常只生成 "路径"，即没有与路径相关的时间信息。`MoveIt` 包含几种轨迹处理算法，可以处理这些路径，并根据施加在各个关节上的最大速度和加速度限制生成适当时间参数化的轨迹。
这些限制是从为每个机器人指定的特殊 `joint_limits.yaml` 配置文件中读取的。配置文件是可选的，它会覆盖 `URDF` 中的任何速度或加速度限制。截至 2023 年 1 月的推荐算法是时间最优轨迹生成算法（`TimeOptimalTrajectoryGeneration，TOTG`）。该算法的一个注意事项是，机器人必须以静止状态开始和结束。默认情况下，`TOTG` 的时间步长为 `0.1` 秒。
###### datetime:2025/02/11 15:01

###### author:nzb

# [时间参数化](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html#time-parameterization)

MoveIt 目前主要是一个运动学运动规划框架——它规划关节或末端执行器的位置，但不规划速度或加速度。然而，MoveIt 确实利用后处理来为运动学轨迹添加时间参数化，以生成速度和加速度值。下面我们将解释 MoveIt 中涉及此部分的设置和组件。

## 速度控制

### 从文件加载

默认情况下，MoveIt 将关节轨迹的关节速度和加速度限制设置为机器人 URDF 或 `joint_limits.yaml` 文件中允许的默认值。`joint_limits.yaml` 文件由 Setup Assistant 生成，最初是 URDF 中值的精确副本。用户可以根据需要修改这些值，使其小于原始 URDF 值。可以通过以下键更改特定关节的属性：`max_position`、`min_position`、`max_velocity`、`max_acceleration`、`max_jerk`。可以通过 `has_velocity_limits`、`has_acceleration_limits`、`has_jerk_limits` 键来启用或禁用关节限制。

### 运行时调整

参数化运动学轨迹的速度也可以在运行时进行调整，作为配置值中设置的最大速度和加速度的一部分，取值范围为 0 到 1。要在每次运动计划的基础上更改速度，可以按照 `MotionPlanRequest.msg` 中的描述设置两个缩放因子。MoveIt MotionPlanning RViz 插件中也提供了用于设置这两个因子的旋钮。

## 时间参数化算法

MoveIt 支持不同的算法来对运动学轨迹进行后处理，以添加时间戳和速度/加速度值。目前 MoveIt 中只有一个选项：

### 时间最优轨迹生成

该方法生成的轨迹具有非常平滑和连续的速度曲线。该方法基于将路径段拟合到原始轨迹，然后从优化后的路径中采样新的路径点。这与严格的时间参数化方法不同，因为生成的路径点可能会在一定的容差范围内偏离原始轨迹。因此，在使用此方法时，可能需要进行额外的碰撞检查。它是 MoveIt 2 中的默认方法。

使用时间参数化算法作为规划请求适配器的用法在[本教程中(使用规划请求适配器)](./06-motion_planning_pipeline.md)有详细说明。

## 抖动限制轨迹平滑
TOTG 等时间参数化算法计算轨迹的速度和加速度，但没有一种时间参数化算法支持抖动限制。 这种情况并不理想--轨迹上的较大颠簸会导致运动生涩或硬件损坏。 作为进一步的后处理步骤，可以应用 Ruckig 运动限制平滑算法来限制轨迹上的关节运动。 

要应用 Ruckig 平滑算法，应在 `joint_limits.yaml` 文件中定义运动限制。 如果没有为任何关节指定挺举限制，将应用合理的默认值并打印警告。

最后，将 Ruckig 平滑算法添加到规划请求适配器列表中（通常在 `ompL_planning.yaml` 中）。Ruckig 平滑算法应最后运行，因此将其放在列表的顶部：

```yaml
response_adapters:
  - default_planning_request_adapters/AddRuckigTrajectorySmoothing
  - default_planning_request_adapters/AddTimeOptimalParameterization
```

---
###### datetime:2025/12/21 16:00

###### author:nzb

# IR-SIM 

""IR-SIM"" 是一款开源、基于 Python 的轻量级机器人仿真器，面向导航、控制与学习场景。它提供简单易用的框架，并内置碰撞检测，用于建模机器人、传感器与环境。IR-SIM 非常适合科研与教学，可在自定义场景下以最少的编码与硬件需求，快速构建机器人与 AI 算法的原型。

## 核心特性

可仿真多种机器人平台，覆盖丰富的运动学模型、传感器与行为

借助简洁的 YAML 文件即可快速配置与自定义仿真场景，无需复杂编码

通过 matplotlib 实时可视化仿真结果，获得即时反馈与分析

支持每个仿真对象的碰撞检测与行为控制

## 链接

- [快速快速-几分钟内即可运行IR-SIM](https://ir-sim.readthedocs.io/zh-cn/stable/get_started/index.html)
- [用户指南-学习如何高效使用IR-SIM](https://ir-sim.readthedocs.io/zh-cn/stable/usage/index.html)
- [配置-YAML 配置文件与示例](https://ir-sim.readthedocs.io/zh-cn/stable/yaml_config/index.html)
- [API参考-完整的API文档](https://ir-sim.readthedocs.io/zh-cn/stable/api/index.html)

## 用户指南目录

- 创建环境
  - Python 脚本与 YAML 配置文件
  - 重要参数说明
  - 基础仿真循环
  - 环境控制与状态
  - 配置环境标题
- 配置机器人与障碍物
  - 机器人配置参数
  - 障碍物配置参数
  - 多机器人/障碍物的高级配置
- 为机器人配置传感器
  - LiDAR 配置参数
- 为对象配置行为
  - 行为配置参数
  - 自定义行为的高级配置
- 配置网格地图环境
  - 网格地图配置参数
- 配置键盘/鼠标控制
  - 键盘控制配置参数
  - 键盘控制按键映射
  - 鼠标控制
- 配置动态随机环境
  - 随机障碍物配置参数
  - 关键参数说明
- 渲染并保存动画
  - 渲染环境
  - 将动画保存为 GIF
  - 将动画保存为视频
  - 3D 绘图

## 基于 IR-SIM 的项目

### 科研项目

- [rl-rvo-nav（RAL & ICRA2023）](https://github.com/hanruihua/rl_rvo_nav)

- [RDA_planner（RAL & IROS2023）](https://github.com/hanruihua/RDA_planner)

  RDA Planner 是一款高性能、基于优化的模型预测控制 (MPC) 运动规划器，专为复杂杂乱环境中的自主导航而设计。RDA 利用交替方向乘子法 (ADMM) 将复杂的优化问题分解为若干个简单的子问题。这种分解方法能够并行计算每个障碍物的避障约束，从而显著提高计算速度。

- [NeuPAN（T-RO 2025）](https://github.com/hanruihua/NeuPAN)

### 深度强化学习项目

- [DRL-robot-navigation-IR-SIM](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM)

- [AutoNavRL](https://github.com/harshmahesheka/AutoNavRL)
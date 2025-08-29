###### datetime:2025/08/18 12:04

###### author:nzb

# legged_gym项目代码阅读

[项目地址](https://github.com/leggedrobotics/legged_gym/tree/master)

## 一、介绍

`legged_gym`提供了用于训练 `ANYmal`（和其他机器人）使用 `NVIDIA` 的 `Isaac Gym` 在崎岖地形上行走的环境。
它包括模拟到真实传输所需的所有组件：执行器网络、摩擦和质量随机化、噪声观测和训练过程中的随机推送。

其中该项目需要用到 `Isaac_gym` （已停止维护）与 `rsl_rl1.2.0` （大版本已不是最新），可能更适合用于学习。

## 二、文件结构

1、总体结构

```text
legged_gym/
├── envs/ 各种腿部机器人环境的定义
├── scripts/ 训练、测试脚本
├── utils/ 实用工具函数和类
├── tests/ 测试训练环境搭建情况的脚本
```

2、各目录详细介绍

**envs/**：定义了不同的腿部机器人环境，每个环境包含一个 `Python` 文件和对应的配置文件。
                        
```plain
envs/
├── cassie/                 cassie 机器人的环境
│  ├── cassie.py            环境实现代码
│  └── cassie_config.py     环境和训练的配置文件
└── ...
```

**scripts/**：包含用于训练、测试的脚本。

```plain
scripts/
├── train.py         		训练脚本，启动强化学习训练过程
├── play.py          		运行训练好的策略进行测试
└── ...
```

**utils/**：包含各种实用工具函数和类，用于辅助开发和调试。

```plain
utils/
├── task_register.py 		任务注册脚本，主要是对envs下的脚本的参数的读取，设置
├── helper.py        		参数初始化脚本，训练环境的设置等
├── logger.py        		log文件操作
├── math.py          		一些数学运算脚本
├── terrain.py       		地形设置脚本
└── ...
```

**tests/**：包含训练环境测试脚本

```plain
tests/
├── test_env.py 			测试训练环境搭建情况的脚本
```

## 三、代码解析

### 1、scripts

#### 1.1 scripts/train.py

```python
import numpy as np                        
import os                                 
from datetime import datetime             

import isaacgym                        
from legged_gym.envs import *              
from legged_gym.utils import get_args, task_registry  # 从 legged_gym.utils 中导入 get_args（解析命令行参数）和 task_registry（任务注册器，用于创建环境和算法运行器）
import torch                              

def train(args):
    # 通过任务注册器根据指定的任务名称和参数创建仿真环境和环境配置
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 通过任务注册器根据指定的任务名称和参数创建强化学习算法运行器（例如 PPO）以及训练配置
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 开始学习训练，传入训练的最大迭代次数和是否在随机的episode长度下初始化环境
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()   # 解析命令行参数，返回包含所有参数的对象
    train(args)         # 调用训练函数，开始训练流程
```

- 详细说明
  - **任务注册器 (task_registry)：**任务注册器是 `legged_gym` 框架中的一个核心组件，它根据任务名称（例如不同的运动任务）创建相应的仿真环境和强化学习算法运行器，从而根据不同任务进行训练。
  - **主程序入口：**当该脚本被直接执行时，会首先调用 `get_args()` 获取命令行参数，然后调用 `train(args)` 开始整个训练流程。


#### 1.2 scripts/test.py

```python
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym  
from legged_gym.envs import *  # 导入所有环境相关的模块和类
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
# 导入工具函数

import numpy as np  
import torch       


def play(args):
    """
    测试/演示函数，用于加载训练好的策略并在仿真环境中运行
    """
    # 获取环境配置（env_cfg）和训练配置（train_cfg）
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 重写部分参数用于测试（主要是降低仿真规模和取消某些随机化/噪声）
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)  # 限制环境实例数量，最多50个
    env_cfg.terrain.num_rows = 5  # 限制地形行数为5
    env_cfg.terrain.num_cols = 5  # 限制地形列数为5
    env_cfg.terrain.curriculum = False  # 禁用地形难度逐步增加
    env_cfg.noise.add_noise = False  # 禁用噪声
    env_cfg.domain_rand.randomize_friction = False  # 禁用摩擦系数随机化
    env_cfg.domain_rand.push_robots = False  # 禁用对机器人施加外部扰动

    # 创建环境，环境配置
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 获取当前环境的观测值
    obs = env.get_observations()
    
    # 设置训练配置，表示恢复之前的训练（resume=True）
    train_cfg.runner.resume = True
    # 创建强化学习算法运行器（例如 PPO）和对应的训练配置
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 从算法运行器中获取推理使用的策略，并指定设备（例如 GPU 或 CPU）
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        # 导出策略，将 actor_critic 模型保存为 JIT 模块
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 创建日志记录器，用于记录仿真过程中的状态和奖励等信息，参数为仿真步长 dt
    logger = Logger(env.dt)
    
    # 设置日志记录的机器人和关节索引，方便后续日志输出（选择记录第0个机器人、第1个关节）
    robot_index = 0  # 指定用于日志记录的机器人索引
    joint_index = 1  # 指定用于日志记录的关节索引
    stop_state_log = 100  # 前100步记录状态数据，用于绘制状态变化曲线
    stop_rew_log = env.max_episode_length + 1  # 在仿真最大步数之后记录奖励信息
    
    # 设置相机初始位置、速度和观察方向，便于动态更新观察视角
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)  # 相机初始位置
    camera_vel = np.array([1., 1., 0.])  # 相机移动速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)  # 计算相机观察方向
    img_idx = 0  # 图像保存时的编号计数器

    # 运行仿真循环，循环次数为 10 倍的最大episode长度
    for i in range(10 * int(env.max_episode_length)):
        # 根据当前观测值使用策略生成动作（actions）
        actions = policy(obs.detach())
        # 执行动作，更新观测值、奖励、是否完成标志以及其他信息
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 如果需要记录视频帧，则每隔一定步数保存当前视图的截图
        if RECORD_FRAMES:
            if i % 2:
                # 构造图片保存路径
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                # 保存当前视图的截图
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        
        # 如果设置移动相机，则根据相机速度更新相机位置，并设置新的观察位置
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # 前 stop_state_log 步记录状态信息
        if i < stop_state_log:
            # 记录各项状态信息，包括关节目标位置、实际位置、速度、施加的力矩、命令、底盘速度和足端接触力等
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        # 当达到 stop_state_log 步时，绘制记录的状态图
        elif i == stop_state_log:
            logger.plot_states()
        
        # 在 0 到 stop_rew_log 之间的步数内，记录奖励信息
        if 0 < i < stop_rew_log:
            # 如果有 episode 信息（即完成的episode），则记录奖励
            if infos["episode"]:
                # 统计重置的 episode 数量
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        # 当达到 stop_rew_log 步时，打印平均奖励信息
        elif i == stop_rew_log:
            logger.print_rewards()


# 主入口：当该脚本作为主程序运行时执行
if __name__ == '__main__':
    EXPORT_POLICY = True   # 是否将训练好的策略导出为 JIT 模块
    RECORD_FRAMES = False  # 是否记录仿真帧（生成视频）
    MOVE_CAMERA = False    # 是否启用相机动态移动
    args = get_args()
    play(args)

```

### 2、utils

#### 2.1 utils/helpers.py

```python
import os                  
import copy               
import torch                
import numpy as np         
import random             
from isaacgym import gymapi         # 导入 Isaac Gym 的核心 API
from isaacgym import gymutil        # 导入 Isaac Gym 的工具函数

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    """
    将一个类实例转换为字典形式。
    如果对象不含有 __dict__ 属性，则直接返回对象本身。
    对于类中的属性，遍历所有不以下划线开头的属性，
    如果属性为列表，则递归调用 class_to_dict 转换每个元素，
    否则直接递归转换属性值。
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        # 忽略以 "_" 开头的私有属性或方法
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            # 如果属性值是列表，则对列表中的每个元素进行转换
            for item in val:
                element.append(class_to_dict(item))
        else:
            # 递归转换属性值
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    """
    根据给定字典更新类的属性。
    遍历字典中的键值对：
      - 如果对象的属性类型为类，则递归更新该属性；
      - 否则直接设置属性值。
    """
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    """
    设置随机种子，保证结果的可重复性。
    如果 seed 为 -1，则随机生成一个种子。
    同时设置 Python 内置随机数、NumPy、PyTorch 以及环境变量的种子。
    """
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    """
    解析仿真参数，初始化 Isaac Gym 仿真参数对象。
    根据命令行参数设置物理引擎相关参数，并且从配置文件中进一步覆盖参数。
    """
    # 创建 Isaac Gym 仿真参数对象
    sim_params = gymapi.SimParams()

    # 根据命令行参数设置物理引擎相关参数
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # 如果配置文件中提供了 sim 选项，则解析配置并覆盖上述参数
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果命令行指定了线程数，并且使用 PhysX，则覆盖线程数设置
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    """
    根据给定的根目录、运行编号和检查点编号构建加载模型的路径。
    - 如果 load_run 为 -1，则加载最新一次运行的结果；
    - 如果 checkpoint 为 -1，则加载最新的模型文件；
    返回最终的模型加载路径。
    """
    try:
        runs = os.listdir(root)
        # TODO: 按日期排序以应对月份变化
        runs.sort()
        if 'exported' in runs:
            runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        # 查找运行目录下所有包含 "model" 字符串的文件
        models = [file for file in os.listdir(load_run) if 'model' in file]
        # 根据文件名排序（格式化字符串保证排序正确）
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """
    根据命令行参数更新环境配置（env_cfg）和训练配置（cfg_train）。
    覆盖配置文件中的相关参数，例如环境数量、随机种子、最大迭代次数、实验名称等。
    """
    # 更新环境配置参数
    if env_cfg is not None:
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    # 更新训练配置参数
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    """
    定义并解析命令行参数，包含任务名称、是否恢复训练、实验名称、设备等信息。
    使用 Isaac Gym 自带的参数解析工具 gymutil.parse_arguments 进行解析。
    """
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "指定任务名称"},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "是否从检查点恢复训练"},
        {"name": "--experiment_name", "type": str,  "help": "实验名称，覆盖配置文件中的设置"},
        {"name": "--run_name", "type": str,  "help": "运行名称，覆盖配置文件中的设置"},
        {"name": "--load_run", "type": str,  "help": "恢复训练时加载的运行名称，-1表示加载最新的运行"},
        {"name": "--checkpoint", "type": int,  "help": "检查点编号，-1表示加载最新的检查点"},
        {"name": "--headless", "action": "store_true", "default": False, "help": "是否强制关闭显示"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "是否使用 Horovod 进行多 GPU 训练"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL 算法使用的设备，如 cpu, cuda:0, cuda:1 等"},
        {"name": "--num_envs", "type": int, "help": "创建的环境数量，优先于配置文件中的设置"},
        {"name": "--seed", "type": int, "help": "随机种子，优先于配置文件中的设置"},
        {"name": "--max_iterations", "type": int, "help": "训练最大迭代次数，优先于配置文件中的设置"},
    ]
    # 解析参数，说明信息为 "RL Policy"
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # 对设备名称进行调整：将模拟设备信息与计算设备ID组合
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    """
    将训练好的策略模型导出为 JIT 模块。
    如果 actor_critic 模型包含记忆单元（例如 LSTM），则使用 PolicyExporterLSTM 进行导出；
    否则直接导出 actor 部分，并保存到指定路径。
    """
    if hasattr(actor_critic, 'memory_a'):
        # 如果模型使用了 LSTM（TODO: 支持 GRU）
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        # 深拷贝模型，并转换到 CPU
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    """
    用于将含有 LSTM 模块的策略模型导出为 JIT 模块。
    该类继承自 torch.nn.Module，包含 actor 模块和 LSTM 记忆单元，
    并维护 LSTM 隐藏状态和细胞状态，用于在推理过程中保持记忆。
    """
    def __init__(self, actor_critic):
        super().__init__()
        # 深拷贝 actor 模块
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        # 深拷贝 LSTM 模型，并转移到 CPU
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        # 注册 LSTM 隐藏状态和细胞状态为缓冲区，初始状态均置零
        self.register_buffer('hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer('cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        """
        前向传播：输入 x 经过 LSTM 模块，更新隐藏状态和细胞状态后，
        再通过 actor 模块生成输出动作。
        """
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        """
        重置 LSTM 的隐藏状态和细胞状态为零，供推理时调用。
        """
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        """
        导出当前模型为 JIT 模块，保存到指定路径下。
        """
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
```

#### 2.2 utils/logger.py

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        # 初始化日志记录器，dt 为仿真步长
        self.state_log = defaultdict(list)  # 用于存储状态数据的字典，默认值为列表
        self.rew_log = defaultdict(list)    # 用于存储奖励数据的字典，默认值为列表
        self.dt = dt                        # 仿真步长
        self.num_episodes = 0               # 记录总的 episode 数量
        self.plot_process = None            # 用于存储绘图进程的引用

    def log_state(self, key, value):
        # 将单个状态数据以 key-value 的形式记录到 state_log 中
        self.state_log[key].append(value)

    def log_states(self, dict):
        # 接收一个字典，将其中所有的状态数据逐个记录
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        # 记录奖励数据，同时将奖励乘以 episode 数量
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes  # 更新总的 episode 数量

    def reset(self):
        # 清空状态和奖励日志
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        # 使用子进程绘制状态曲线图
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        # 内部方法：绘制状态数据图
        nb_rows = 3  # 图表的行数
        nb_cols = 3  # 图表的列数
        fig, axs = plt.subplots(nb_rows, nb_cols)  # 创建 3x3 的子图
        for key, value in self.state_log.items():
            # 计算时间轴数据：根据数据长度和仿真步长生成时间序列
            time = np.linspace(0, len(value) * self.dt, len(value))
            break  # 这里只需要一组时间数据，所以跳出循环
        log = self.state_log

        # 绘制关节目标位置与实际测量位置
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()

        # 绘制关节速度
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()

        # 绘制底盘 x 方向线速度与指令速度
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]:
            a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()

        # 绘制底盘 y 方向线速度与指令速度
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]:
            a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()

        # 绘制底盘偏航角速度与指令速度
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()

        # 绘制底盘 z 方向线速度
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()

        # 绘制接触力（z 方向）的变化
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            # 针对每个接触点绘制接触力曲线
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()

        # 绘制关节扭矩与速度的散点图
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()

        # 绘制关节扭矩随时间的变化
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()

        # 显示绘制的图形
        plt.show()

    def print_rewards(self):
        # 打印每秒平均奖励
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        # 析构函数：如果绘图进程存在，则将其终止
        if self.plot_process is not None:
            self.plot_process.kill()

```

#### 2.3 utils/math.py

```python
# 将四元数限制为只包含偏航角分量，并用该四元数旋转向量
def quat_apply_yaw(quat, vec):
    # 复制四元数，并将其重塑为形状为(-1, 4)
    quat_yaw = quat.clone().view(-1, 4)
    # 将四元数的前两个分量（对应 roll 和 pitch）置为0，只保留 yaw 分量和标量部分
    quat_yaw[:, :2] = 0.
    # 对四元数进行归一化
    quat_yaw = normalize(quat_yaw)
    # 使用处理后的四元数对向量进行旋转
    return quat_apply(quat_yaw, vec)

# 将角度值规范化到 [-π, π] 区间内
def wrap_to_pi(angles):
    # 将角度取模 2π，得到[0, 2π)范围内的值
    angles %= 2 * np.pi
    # 对大于 π 的角度，减去 2π，使其落在[-π, π)范围内
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

# 生成在给定范围 [lower, upper] 内的随机浮点数张量，分布经过平方根变换
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    # 生成形状为 shape 的随机张量，其元素均匀分布在 [-1, 1] 之间
    r = 2 * torch.rand(*shape, device=device) - 1
    # 对随机数进行分段处理：
    # 如果 r 小于 0，则取 -sqrt(-r)；如果 r 大于等于0，则取 sqrt(r)
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    # 将 r 从 [-1, 1] 映射到 [0, 1]
    r = (r + 1.) / 2.
    # 根据 [lower, upper] 范围对 r 进行线性缩放，返回最终结果
    return (upper - lower) * r + lower

```

#### 2.4 utils/task_registry.py

```python
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        # 初始化任务注册器，存储任务类、环境配置和训练配置
        self.task_classes = {}  # 存储任务名称与对应环境类的字典
        self.env_cfgs = {}      # 存储任务名称与对应环境配置的字典
        self.train_cfgs = {}    # 存储任务名称与对应训练配置的字典
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        # 注册一个任务，将任务名称、对应的环境类、环境配置和训练配置存入字典中
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        # 根据任务名称返回对应的环境类
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        # 根据任务名称返回对应的环境配置和训练配置
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 将训练配置中的随机种子复制到环境配置中
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """
        根据任务名称或提供的配置文件创建环境

        Args:
            name (string): 注册的环境名称
            args (Args, optional): Isaac Gym 命令行参数，如果为 None 则调用 get_args() 获取。默认值为 None。
            env_cfg (Dict, optional): 用于覆盖注册配置的环境配置文件。默认值为 None。

        Raises:
            ValueError: 如果没有找到与 'name' 对应的已注册环境，则抛出异常

        Returns:
            VecEnv: 创建的环境实例
            LeggedRobotCfg: 对应的环境配置
        """
        # 如果未传入命令行参数，则调用 get_args() 获取参数
        if args is None:
            args = get_args()
        # 检查是否有对应名称的已注册环境
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # 加载配置文件
            env_cfg, _ = self.get_cfgs(name)
        # 根据命令行参数覆盖配置文件中的相关参数
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # 设置随机种子，保证实验可重复性
        set_seed(env_cfg.seed)
        # 解析仿真参数：先将 sim 配置转换为字典，再调用 parse_sim_params
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # 根据配置创建环境实例
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """
        根据任务名称或提供的训练配置创建强化学习算法运行器

        Args:
            env (VecEnv): 用于训练的环境实例
            name (string, optional): 注册的环境名称。如果为 None，则使用配置文件中的训练配置。默认值为 None。
            args (Args, optional): Isaac Gym 命令行参数，如果为 None 则调用 get_args() 获取。默认值为 None。
            train_cfg (Dict, optional): 训练配置文件。如果为 None，则根据 'name' 加载配置文件。默认值为 None。
            log_root (str, optional): Tensorboard 日志目录。设置为 None 可避免日志记录（例如测试时）。
                                      默认值为 "default"，即日志保存路径为 <LEGGED_GYM_ROOT_DIR>/logs/<experiment_name>。

        Raises:
            ValueError: 如果 'name' 和 'train_cfg' 均为 None，则抛出异常
            Warning: 如果同时提供了 'name' 和 'train_cfg'，则忽略 'name'

        Returns:
            OnPolicyRunner: 创建的强化学习算法运行器
            LeggedRobotCfgPPO: 对应的训练配置
        """
        # 如果未传入命令行参数，则调用 get_args() 获取参数
        if args is None:
            args = get_args()
        # 如果未传入训练配置，则根据任务名称加载配置文件
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 加载训练配置文件
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # 根据命令行参数覆盖训练配置中的相关参数
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 根据 log_root 参数构造日志保存目录
        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 将训练配置转换为字典
        train_cfg_dict = class_to_dict(train_cfg)
        # 创建强化学习算法运行器
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        # 如果设置为恢复训练，则加载之前保存的模型
        resume = train_cfg.runner.resume
        if resume:
            # 获取模型加载路径
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# 全局创建任务注册器实例
task_registry = TaskRegistry()
```

#### 2.5 utils/terrain.py

```python
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        # 初始化地形类，cfg 为地形配置，num_robots 为机器人数量
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type  # 地形类型，来自配置中的 mesh_type
        if self.type in ["none", 'plane']:
            # 如果地形类型为 "none" 或 "plane"，则无需进一步处理
            return
        self.env_length = cfg.terrain_length  # 环境的长度
        self.env_width = cfg.terrain_width    # 环境的宽度
        # 计算每种地形类型的累积比例，用于随机生成地形时选择不同类型
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # 计算子地形数量，等于行数乘以列数
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 初始化每个子环境的原点坐标（3D坐标）
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 根据水平缩放计算每个子环境在像素级别的宽度和长度
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # 计算边界大小（转换为像素单位）
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        # 总的列数：所有子环境的像素宽度加上左右边界
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # 总的行数：所有子环境的像素长度加上上下边界
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 初始化原始高度场为全零数组，数据类型为 int16
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        # 根据配置决定使用哪种地形生成方法
        if cfg.curriculum:
            self.curiculum()  # 使用难度递增的地形生成方法
        elif cfg.selected:
            self.selected_terrain()  # 使用预选的地形生成方法
        else:    
            self.randomized_terrain()  # 随机生成地形
        
        # 保存生成的高度场样本
        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            # 如果地形类型为三角网格，转换高度场为顶点和三角形信息
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )
    
    def randomized_terrain(self):
        # 随机生成地形，每个子地形均生成随机参数的地形
        for k in range(self.cfg.num_sub_terrains):
            # 将子地形索引 k 转换为二维网格的行、列索引
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            # 随机生成一个介于 0 到 1 之间的数，用于选择地形类型
            choice = np.random.uniform(0, 1)
            # 随机选择一个难度值（示例：0.5, 0.75 或 0.9）
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # 根据随机值和难度生成具体的地形
            terrain = self.make_terrain(choice, difficulty)
            # 将生成的地形添加到整体高度场中，位置由 i, j 决定
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        # 生成按照难度逐渐递增的地形（课程式训练）
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 难度与行索引成正比
                difficulty = i / self.cfg.num_rows
                # choice 与列索引成正比，稍作偏移避免为零
                choice = j / self.cfg.num_cols + 0.001
                # 生成地形
                terrain = self.make_terrain(choice, difficulty)
                # 添加地形到整体高度场中
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        # 根据预选配置生成特定类型的地形
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # 将子地形索引转换为二维行、列索引
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            # 创建一个子地形实例
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )
            # 通过 eval 动态调用预选的地形生成函数，传入参数
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            # 添加生成的地形到整体高度场中
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        # 根据随机选择值和难度生成单个子地形
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        # 根据难度计算坡度、台阶高度、离散障碍物高度、石块尺寸、石块间距、缺口尺寸和坑洞深度
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        # 根据 choice 值与累积比例决定生成哪种类型的地形
        if choice < self.proportions[0]:
            # 前半部分地形，如果 choice 小于累积比例的一半，则反转坡度
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            # 生成金字塔式坡道地形，并在其上增加随机高度扰动
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            # 生成阶梯地形，如果 choice 小于第三个比例，则反转台阶高度
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            # 生成离散障碍物地形
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.
            )
        elif choice < self.proportions[5]:
            # 生成踏石地形
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.,
                platform_size=4.
            )
        elif choice < self.proportions[6]:
            # 生成缺口地形
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            # 生成坑洞地形
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        # 将生成的子地形添加到整体高度场中对应的位置
        i = row
        j = col
        # 根据边界和子环境像素尺寸计算该子环境在整体高度场中的起始和结束索引
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        # 将子地形的高度场复制到整体高度场的对应区域
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 计算每个子环境的原点坐标
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # 计算子环境中心区域对应在地形高度场中的索引
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        # 计算该子环境的原点高度（取中心区域高度场的最大值，并按垂直缩放）
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    # 生成缺口地形：在地形中心挖一个缺口，并在中间留出一个平台
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    # 计算中心位置
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    # 计算缺口区域的索引范围
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
    # 将缺口区域高度设为非常低（-1000）以形成缺口
    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    # 将中心平台区域高度设为 0
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    # 生成坑洞地形：在地形中心生成一个坑洞，深度根据 depth 设定
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    # 计算中心平台区域的索引范围
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    # 将中心区域的高度设置为 -depth，以形成坑洞
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

```


### 3、envs

#### 3.1 envs/base/legged_robot.py

- 注意为为几大块
  - 1. 环境初始化
  - 2. 创建机器人
  - 3. 探索
  - 4. 进步
  - 5. 奖励

```python
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):

    # ------------- env init --------------
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """解析传入的配置文件，
           调用 create_sim()（该函数创建模拟、地形和环境），
           初始化训练过程中使用的 pytorch 缓冲区

        参数:
            cfg (Dict): 环境配置文件
            sim_params (gymapi.SimParams): 模拟参数
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX（必须是 PhysX）
            sim_device (string): 'cuda' 或 'cpu'
            headless (bool): 如果为 True，则不进行渲染运行
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def _init_buffers(self):
        """初始化将包含模拟状态和处理后数据的 torch 张量"""
        # 获取 gym 的 GPU 状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 为不同部分创建包装张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # 形状：num_envs, num_bodies, xyz轴

        # 初始化后续使用的数据
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x 速度, y 速度, 偏航速度, 朝向
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO 修改此处
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # 关节初始位置偏移和 PD 增益
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"关节 {name} 的 PD 增益未定义，设置为零")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )

    def create_sim(self):
        """创建模拟、地形和环境"""
        self.up_axis_idx = 2  # 对于 z 轴为 2，y 轴为 1 —— 根据重力方向调整
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "地形网格类型无法识别。允许的类型为 [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """设置摄像头的位置和朝向"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _draw_debug_vis(self):
        """绘制调试用可视化（会大幅降低模拟速度）。
        默认行为：绘制高度测量点
        """
        # 绘制高度线
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(
                    self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]
                )
                .cpu()
                .numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )

    def _create_ground_plane(self):
        """向模拟中添加一个地面平面，根据配置设置摩擦系数和反弹系数。"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """向模拟中添加一个高度场地形，根据配置设置参数。"""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """向模拟中添加一个三角网格地形，根据配置设置参数。"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _init_height_points(self):
        """返回采样高度测量点（在底盘坐标系下）的坐标

        返回:
            [torch.Tensor]: 形状为 (num_envs, self.num_height_points, 3) 的张量
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    # ------------- create robot --------------
    def _process_rigid_shape_props(self, props, env_id):
        """回调函数，允许存储/更改/随机化每个环境中刚体形状的属性。
           在环境创建时调用。
           默认行为：随机化每个环境的摩擦系数

        参数:
            props (List[gymapi.RigidShapeProperties]): 资产中每个形状的属性列表
            env_id (int): 环境编号

        返回:
            [List[gymapi.RigidShapeProperties]]: 修改后的刚体形状属性列表
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # 准备摩擦系数随机化
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """回调函数，允许存储/更改/随机化每个环境中自由度（DOF）的属性。
           在环境创建时调用。
           默认行为：存储 URDF 中定义的位置、速度和扭矩限制

        参数:
            props (numpy.array): 资产中每个自由度的属性
            env_id (int): 环境编号

        返回:
            [numpy.array]: 修改后的自由度属性
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # 软限制
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        # 如果 env_id==0:
        #     累加每个部分的质量并打印各部分质量（随机化之前）
        #     print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # 随机化底盘质量
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _create_envs(self):
        """创建环境：
        1. 加载机器人 URDF/MJCF 资产，
        2. 对于每个环境：
           2.1 创建环境实例，
           2.2 调用自由度和刚体属性的回调函数，
           2.3 使用这些属性创建角色并将其加入环境
        3. 存储机器人的各部分索引信息
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 保存资产中的刚体名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # 创建环境实例
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _get_env_origins(self):
        """设置环境原点。对于粗糙地形，原点由地形平台定义；
        否则创建一个网格布局。
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # 将机器人放置在地形定义的原点处
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # 创建机器人网格
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _get_heights(self, env_ids=None):
        """在机器人周围采样地形高度。
           这些点相对于底盘位置并经过底盘偏航旋转

        参数:
            env_ids (List[int], optional): 指定的环境编号子集，默认为 None

        返回:
            [type]: 返回采样到的高度
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("无法在 'none' 类型的地形中测量高度")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # ------------- explore --------------
    def _push_robots(self):
        """随机推机器人，模拟施加冲量，通过设置随机的底盘速度实现"""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # 线速度 x/y
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def _update_terrain_curriculum(self, env_ids):
        """实现基于游戏设计的地形课程

        参数:
            env_ids (List[int]): 需要重置的环境编号列表
        """
        # 实现地形课程
        if not self.init_done:
            # 初始重置时不改变
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # 行走距离足够远的机器人提升到更难的地形
        move_up = distance > self.terrain.env_length / 2
        # 行走距离不足一半的机器人降级到更简单的地形
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, :2], dim=1)
            * self.max_episode_length_s
            * 0.5
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # 当机器人达到最高级别时，将其分配到随机级别
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # 最低级别为 0
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        """实现逐步增加指令幅度的课程

        参数:
            env_ids (List[int]): 需要重置的环境编号列表
        """
        # 如果追踪奖励达到最大值的 80%，则扩大指令范围
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    def _get_noise_scale_vec(self, cfg):
        """设置用于对观测添加噪声的缩放向量
           [注意]: 当更改观测结构时，需要对该函数进行适配

        参数:
            cfg (Dict): 环境配置文件

        返回:
            [torch.Tensor]: 用于乘以 [-1, 1] 均匀分布的缩放向量
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # 指令部分
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0  # 前一次动作
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    def check_termination(self):
        """
        检查是否有环境需要重置,保护真机
        """
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # 超时不计终止奖励
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """重置指定环境
           调用 self._reset_dofs(env_ids)、self._reset_root_states(env_ids) 和 self._resample_commands(env_ids)
           [可选] 调用 self._update_terrain_curriculum(env_ids)、self.update_command_curriculum(env_ids) 并记录本回合信息
           重置一些缓冲区

        参数:
            env_ids (list[int]): 需要重置的环境编号列表
        """
        if len(env_ids) == 0:
            return
        # 更新课程（curriculum）
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # 避免每步都更新指令课程，因为最大指令对所有环境通用
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)

        # 重置机器人状态
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # 重置缓冲区
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # 填充额外信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # 记录额外的课程信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][
                1
            ]
        # 将超时信息发送给算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        """重置选定环境的自由度位置和速度
           位置在默认位置的 0.5 到 1.5 倍之间随机选择，
           速度则设为零。

        参数:
            env_ids (List[int]): 环境编号列表
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """重置选定环境的根状态（位置和速度）
           根据课程（curriculum）设置底盘位置，
           并在 -0.5 到 0.5 范围内为底盘速度随机选择初值

        参数:
            env_ids (List[int]): 环境编号列表
        """
        # 底盘位置
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy 方向在中心 1 米范围内随机偏移
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # 底盘速度
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: 线速度, [10:13]: 角速度
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        """为部分环境随机选择新的指令

        参数:
            env_ids (List[int]): 需要生成新指令的环境编号列表
        """
        # 最小值和最大值中间随机取一个值
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # 将较小的指令设为零，随机到了一个不大于0.2的值，则将值设为零，避免机器人不知道怎么走，正常来说不会这么速度走，踱步就是另一种步态，需要另外训练
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    # ------------- progress --------------
    def step(self, actions):
        """应用动作，执行模拟，并调用 self.post_physics_step()

        参数:
            actions (torch.Tensor): 形状为 (num_envs, num_actions_per_env) 的张量
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 执行物理步进并渲染每一帧
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # 返回剪裁后的观测、状态（None）、奖励、结束标志以及额外信息
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        """检查是否需要结束，计算观测和奖励
        调用 self._post_physics_step_callback() 来进行一些通用计算，
        如果需要则调用 self._draw_debug_vis() 绘制调试可视化
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1  # 作为一个时间步来存储的,就是每个这个智能体,每个机器人他经过了多少个step,就是用这个数来算
        self.common_step_counter += 1

        # 准备相关量
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        self._post_physics_step_callback()

        # 计算观测、奖励、重置标志等
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # 在某些情况下可能需要一次模拟步以刷新某些观测值（例如身体位置）

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """在计算终止、奖励和观测之前调用的回调函数
        默认行为：基于目标和朝向计算角速度指令，计算测量到的地形高度，以及随机推机器人
        """
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _compute_torques(self, actions):
        """根据动作计算扭矩。
           动作可以被解释为 PD 控制器给出的目标位置或速度，或直接作为缩放后的扭矩值。
           [注意]: 扭矩张量的维度必须与自由度数量相同，即使某些自由度没有被驱动。

        参数:
            actions (torch.Tensor): 动作

        返回:
            [torch.Tensor]: 传递给模拟的扭矩张量
        """
        # PD 控制器
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"未知的控制器类型: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        """计算观测值"""
        # 这些值通常都是用来计算奖励,或者说计算这个action用的,这就是observation观测值
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        # 如果不盲目，则添加感知输入
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # 如果需要则添加噪声
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    # ------------- reward --------------
    def _prepare_reward_function(self):
        """准备奖励函数列表，这些函数将被调用以计算总奖励。
        查找名称为 self._reward_<REWARD_NAME> 的函数，其中 <REWARD_NAME> 为配置中所有非零奖励比例的名称。
        """
        # 移除比例为零的奖励，并将非零奖励乘以 dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # 准备函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # 初始化每回合的奖励累积和
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def compute_reward(self):
        """计算奖励
        调用所有非零权重的奖励函数（在 self._prepare_reward_function() 中处理过），
        将各项奖励加到本回合累积奖励和总奖励上
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # 在剪裁后添加终止奖励
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # ------------ 奖励函数 ----------------
    # 1、Task 任务类奖励，例如： _reward_tracking_lin_vel，希望达到恒定速度
    # 2、Smooth 平滑奖励，例如： _reward_action_rate
    # 3、Safety 安全奖励，例如： _reward_dof_pos_limits
    # 4、Beauty 美观奖励，不抬腿走，拖着腿走，例如： _reward_feet_air_time
    def _reward_lin_vel_z(self):
        # 惩罚底盘在 z 轴上的线速度
        # 然后配置的值是-2.0,如果是这个算出来的值是零,他奖励就是零,如果说你的线速度越大,他这个值就越大,然后你这个负的就越大,所以说他是一个惩罚,就是一个负奖励
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # 惩罚底盘在 xy 轴上的角速度
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # 惩罚底盘姿态不水平
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # 惩罚底盘高度偏离目标值
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # 惩罚扭矩过大
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # 惩罚自由度速度过大
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # 惩罚自由度加速度过大
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_action_rate(self):
        # 惩罚动作变化率过大
        # smooth 平滑奖励，使动作变化率变小
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # 对指定部位的碰撞进行惩罚
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        # 终止奖励/惩罚
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # 惩罚自由度位置接近极限的情况
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # 下限
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # 惩罚自由度速度接近极限的情况
        # 为避免过大惩罚，将每个关节的最大误差裁剪为 1 rad/s
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        # 惩罚扭矩接近极限的情况
        return torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # 对线速度指令（xy轴）的追踪奖励
        # 任务类奖励，差值越小奖励越大
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # 对角速度指令（偏航）的追踪奖励
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # 奖励长步伐
        # 需要过滤接触，因为 PhysX 在网格上报告的接触不太可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # 仅在首次接触地面时给予奖励
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # 对于零指令不给予奖励
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # 惩罚足部撞击垂直表面
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        # 对于零指令，惩罚机器人运动
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_feet_contact_forces(self):
        # 惩罚足部接触力过大
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )
```

#### 3.2 envs/base/legged_robot_config.py

```python
from .base_config import BaseConfig

# 定义四足机器人仿真配置类
class LeggedRobotCfg(BaseConfig):
    # 环境参数设置
    class env:
        num_envs = 4096                    # 并行仿真环境数量
        num_observations = 235             # 观测向量的维度
        num_privileged_obs = None          # 如果不为 None，则 step() 返回特权观测（用于不对称训练），否则返回 None
        num_actions = 12                   # 动作维度（机器人控制信号数量），机器人可以动的关节，不能是 fexed joints
        env_spacing = 3.                   # 环境之间的间隔（对于高度场/三角网格不使用）
        send_timeouts = True               # 是否将超时信息发送给训练算法
        episode_length_s = 20              # 每个回合的时长（秒），如果说这个如果说你这个设置的没有设置,或者设置的比较长的话,他那个机器人可能会采取一种就是不动的状态,他不接受你的命令,他就保持不动,所以说如果你手动设置一个时间,它就会在这个时间固定被重置,确保它能够不摆烂

    # 地形参数设置
    class terrain:
        mesh_type = 'trimesh'              # 地形网格类型，可选：none, plane, heightfield 或 trimesh
        horizontal_scale = 0.1             # 地形水平缩放比例（单位：米）
        vertical_scale = 0.005             # 地形垂直缩放比例（单位：米）
        border_size = 25                   # 地形边界尺寸（单位：米）
        curriculum = True                  # 是否使用地形难度课程（逐步增加难度）
        static_friction = 1.0              # 静摩擦系数
        dynamic_friction = 1.0             # 动态摩擦系数
        restitution = 0.                   # 反弹系数（弹性系数）
        # 针对粗糙地形的设置
        measure_heights = True             # 是否测量地形高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                                          # 在 x 轴上采样高度测量点（构成 1m×1.6m 区域，中心线除外）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
                                          # 在 y 轴上采样高度测量点
        selected = False                   # 是否选择唯一的地形类型，并传递所有参数
        terrain_kwargs = None              # 选定地形的参数字典
        max_init_terrain_level = 5         # 初始地形课程状态的最大难度等级
        terrain_length = 8.                # 地形长度（单位：米）
        terrain_width = 8.                 # 地形宽度（单位：米）
        num_rows = 10                      # 地形行数（不同难度级别数）
        num_cols = 20                      # 地形列数（不同类型数）
        # 地形类型的比例分布：[平缓斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh 专用参数：斜坡阈值，超过该值的斜坡会被修正为垂直表面
        slope_treshold = 0.75

    # 控制指令参数设置
    class commands:
        curriculum = False               # 是否使用指令课程（动态调整指令难度）
        max_curriculum = 1.              # 指令课程中指令的最大幅度
        num_commands = 4                 # 指令维度，默认顺序为：线速度x, 线速度y, 角速度（偏航），以及朝向（当使用朝向模式时，角速度会通过朝向误差计算）
        resampling_time = 10.            # 指令重新采样的时间间隔（秒），重新给命令的时间，一会向左，一会向右，一会向前，一会向后这样
        heading_command = True           # 是否使用朝向指令（True 表示根据朝向误差计算角速度指令）
        class ranges:
            lin_vel_x = [-1.0, 1.0]      # 线速度 x 轴范围（单位：米/秒）
            lin_vel_y = [-1.0, 1.0]      # 线速度 y 轴范围（单位：米/秒）
            ang_vel_yaw = [-1, 1]        # 偏航角速度范围（单位：弧度/秒）
            heading = [-3.14, 3.14]      # 朝向范围（单位：弧度）

    # 初始状态参数设置
    class init_state:
        pos = [0.0, 0.0, 1.]              # 机器人初始位置 [x, y, z]（单位：米），一般 z 设置成机器人高度，不要低于机器人，也不要设置太高
        rot = [0.0, 0.0, 0.0, 1.0]          # 初始旋转，四元数格式 [x, y, z, w]
        lin_vel = [0.0, 0.0, 0.0]           # 初始线速度 [x, y, z]（单位：米/秒）
        ang_vel = [0.0, 0.0, 0.0]           # 初始角速度 [x, y, z]（单位：弧度/秒）
        default_joint_angles = {          # 当动作为 0 时，各关节的目标角度
            "joint_a": 0., 
            "joint_b": 0.}

    # 控制器参数设置
    class control:
        control_type = 'P'               # 控制类型：'P' 表示位置控制，'V' 表示速度控制，'T' 表示直接施加扭矩
        # PD 控制参数
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # 各关节的刚度（单位：N*m/rad）
        damping = {'joint_a': 1.0, 'joint_b': 1.5}       # 各关节的阻尼（单位：N*m*s/rad）
        action_scale = 0.5               # 动作缩放因子，目标角度 = action_scale * 动作 + 默认角度
        decimation = 4                   # 每个策略周期内，控制动作更新的次数（子步数）

    # 机器人模型（资产）参数设置
    class asset:
        file = ""                        # 机器人模型文件路径（需由 LEGGED_GYM_ROOT_DIR 补全）
        name = "legged_robot"            # 机器人角色名称
        foot_name = "None"               # 机器人足部的名称，用于索引接触力张量
        penalize_contacts_on = []        # 用于施加碰撞惩罚的部位列表
        terminate_after_contacts_on = [] # 发生碰撞后终止回合的部位列表
        disable_gravity = False          # 是否禁用重力
        collapse_fixed_joints = True     # 是否合并固定关节连接的刚体（可以通过特定设置保留部分固定关节）
        fix_base_link = False            # 是否固定机器人的底盘
        default_dof_drive_mode = 3       # 默认自由度驱动模式（参考 GymDofDriveModeFlags）
        self_collisions = 0              # 自碰撞设置：0 启用，1 禁用
        replace_cylinder_with_capsule = True  # 是否将碰撞圆柱体替换为胶囊体，提高仿真稳定性
        flip_visual_attachments = True   # 是否翻转视觉附件（部分 .obj 网格需从 y-up 转为 z-up）
        
        density = 0.001                  # 模型密度
        angular_damping = 0.             # 角阻尼
        linear_damping = 0.              # 线阻尼
        max_angular_velocity = 1000.     # 最大角速度
        max_linear_velocity = 1000.      # 最大线速度
        armature = 0.                    # 关节的惯性补偿
        thickness = 0.01                 # 物体厚度

    # 域随机化设置
    class domain_rand:
        randomize_friction = True        # 是否随机化摩擦系数
        friction_range = [0.5, 1.25]       # 摩擦系数随机范围
        randomize_base_mass = False      # 是否随机化底盘质量
        added_mass_range = [-1., 1.]       # 增加质量的随机范围
        push_robots = True               # 是否随机施加外力推动机器人
        push_interval_s = 15             # 推动间隔时间（秒）
        max_push_vel_xy = 1.             # 推动时最大速度（xy 方向）

    # 奖励函数相关设置
    class rewards:
        class scales:
            termination = -0.0         # 终止奖励的比例
            tracking_lin_vel = 1.0       # 追踪线速度奖励比例
            tracking_ang_vel = 0.5       # 追踪角速度奖励比例
            lin_vel_z = -2.0           # 惩罚底盘 z 轴线速度的比例
            ang_vel_xy = -0.05         # 惩罚底盘 xy 轴角速度的比例
            orientation = -0.           # 惩罚底盘姿态偏离水平的比例
            torques = -0.00001         # 惩罚施加扭矩的比例
            dof_vel = -0.              # 惩罚自由度速度的比例
            dof_acc = -2.5e-7          # 惩罚自由度加速度的比例
            base_height = -0.          # 惩罚底盘高度偏离目标的比例
            feet_air_time =  1.0       # 奖励足部空中时间（长步）的比例
            collision = -1.            # 惩罚碰撞的比例
            feet_stumble = -0.0        # 惩罚足部绊倒的比例
            action_rate = -0.01        # 惩罚动作变化率过大的比例
            stand_still = -0.          # 惩罚在零指令下机器人运动的比例

        only_positive_rewards = True    # 如果为 True，则总奖励为负时剪裁为零（避免训练早期出现过大负奖励导致终止）
        tracking_sigma = 0.25            # 追踪奖励的 sigma 参数，用于计算 exp(-error^2 / sigma)
        soft_dof_pos_limit = 1.          # 自由度位置软限制百分比，超出此比例的将受到惩罚
        soft_dof_vel_limit = 1.          # 自由度速度软限制百分比
        soft_torque_limit = 1.           # 扭矩软限制百分比
        base_height_target = 1.          # 底盘目标高度（单位：米）
        max_contact_force = 100.         # 超过该值的接触力将受到惩罚

    # 归一化及剪裁设置
    class normalization:
        class obs_scales:
            lin_vel = 2.0            # 线速度观测缩放系数
            ang_vel = 0.25           # 角速度观测缩放系数
            dof_pos = 1.0            # 自由度位置观测缩放系数
            dof_vel = 0.05           # 自由度速度观测缩放系数
            height_measurements = 5.0  # 高度测量观测缩放系数
        clip_observations = 100.       # 观测值的剪裁上限
        clip_actions = 100.            # 动作值的剪裁上限

    # 噪声设置
    class noise:
        add_noise = True               # 是否在观测中添加噪声
        noise_level = 1.0              # 噪声等级，作为其它噪声尺度的乘数因子
        class noise_scales:
            dof_pos = 0.01           # 自由度位置噪声尺度
            dof_vel = 1.5            # 自由度速度噪声尺度
            lin_vel = 0.1            # 线速度噪声尺度
            ang_vel = 0.2            # 角速度噪声尺度
            gravity = 0.05           # 重力噪声尺度
            height_measurements = 0.1  # 高度测量噪声尺度

    # 观察器（摄像头）设置
    class viewer:
        ref_env = 0                    # 摄像头参考的环境编号
        pos = [10, 0, 6]               # 摄像头位置（单位：米）
        lookat = [11., 5, 3.]           # 摄像头目标位置（单位：米）

    # 模拟器设置
    class sim:
        dt = 0.005                     # 模拟时间步长（秒），仿真的频率，这个值乘以 control.decimation 它就是一个 step 的时间，即 0.02秒，如果你的电脑比较差，可以把 dt 调大点
        substeps = 1                   # 每个时间步内的子步数
        gravity = [0., 0., -9.81]       # 重力加速度（单位：米/秒²）
        up_axis = 1                    # 上升轴：0 表示 y 轴，1 表示 z 轴

        # PhysX 物理引擎设置
        class physx:
            num_threads = 10         # 使用的线程数
            solver_type = 1          # 解算器类型：0 为 pgs，1 为 tgs
            num_position_iterations = 4  # 位置求解器的迭代次数
            num_velocity_iterations = 0  # 速度求解器的迭代次数
            contact_offset = 0.01      # 接触偏移（米）
            rest_offset = 0.0          # 静止偏移（米）
            bounce_threshold_velocity = 0.5  # 反弹阈值速度（米/秒）
            max_depenetration_velocity = 1.0   # 最大去穿透速度（米/秒）
            max_gpu_contact_pairs = 2**23      # GPU 接触对的最大数量（适用于大规模环境）
            default_buffer_size_multiplier = 5
            contact_collection = 2     # 接触数据收集方式：0 表示从不收集，1 表示仅收集最后一个子步，2 表示收集所有子步

# 定义用于 PPO 算法的配置类
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1                          # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 训练运行器类名称
    class policy:
        init_noise_std = 1.0          # 策略初始动作噪声标准差
        actor_hidden_dims = [512, 256, 128]   # Actor 网络隐藏层尺寸
        critic_hidden_dims = [512, 256, 128]    # Critic 网络隐藏层尺寸
        activation = 'elu'            # 激活函数类型（可选：elu, relu, selu, crelu, lrelu, tanh, sigmoid）
        # 针对 ActorCriticRecurrent 可选设置：
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # 训练参数
        value_loss_coef = 1.0          # 值函数损失系数
        use_clipped_value_loss = True  # 是否使用剪裁值函数损失
        clip_param = 0.2               # PPO 剪裁参数
        entropy_coef = 0.01            # 熵正则化系数，用于鼓励探索
        num_learning_epochs = 5        # 每次策略更新的训练轮数
        num_mini_batches = 4           # 小批次数量（每个小批次样本数 = num_envs * nsteps / num_mini_batches）
        learning_rate = 1.e-3          # 学习率
        schedule = 'adaptive'          # 学习率调度方式：'adaptive' 或 'fixed'
        gamma = 0.99                   # 折扣因子
        lam = 0.95                     # GAE（广义优势估计）中的 lambda 参数
        desired_kl = 0.01              # 期望的 KL 散度，用于调节更新步长
        max_grad_norm = 1.             # 最大梯度范数，防止梯度爆炸

    class runner:
        policy_class_name = 'ActorCritic'  # 策略类名称
        algorithm_class_name = 'PPO'         # 算法类名称
        num_steps_per_env = 24         # 每个环境每次迭代采样步数
        max_iterations = 1500          # 策略更新的总次数

        # 日志记录设置
        save_interval = 50             # 每隔多少次迭代保存模型
        experiment_name = 'test'       # 实验名称
        run_name = ''                  # 运行名称
        # 加载和恢复设置
        resume = False                 # 是否从上一次训练中恢复
        load_run = -1                  # -1 表示加载最近一次训练记录
        checkpoint = -1                # -1 表示加载最近保存的模型
        resume_path = None             # 恢复训练的路径（由 load_run 和 checkpoint 确定）

```

## 四、训练过程

## 配置训练参数

- 在 legged_gym/envs/<robot_name>通过配置文件或者命令行参数指定任务名称、环境参数、训练超参数等。
- 例如，可以在命令行中执行：`python train.py --task your_robot_task --other_arg value`
  - 其中 your_robot_task 是你要训练的机器人任务名称。
- 如果需要调整环境规模、随机性、噪声、摩擦系数等参数，可以修改对应的配置文件，或者在代码中进行覆盖。

## 启动训练
  - 运行训练脚本（如 `train.py`）开始训练：

    ```python
    if __name__ == '__main__':
        args = get_args()
        train(args)
    ```
    
    - 训练过程中，程序会按照训练配置（例如 `train_cfg.runner.max_iterations` 指定的迭代次数）不断更新策略，并与环境交互。

## 保存训练结果

  - 在训练过程中，训练日志和模型参数会自动保存到项目的日志目录下。通常日志路径为： 
    - `legged_gym/logs/<experiment_name>/`
    - 其中 `<experiment_name>` 是在配置文件中定义的实验名称。
  - 保存的内容一般包括：
    - 模型检查点：保存训练过程中或最终的策略模型（通常为 `.pt` 或 `.pth` 文件），方便后续恢复训练或进行推理。
    - 训练日志：记录每个 `episode` 的奖励、状态变化、以及其他调试信息。
    - 可视化文件：例如状态曲线图、视频帧截图等（如果设置了记录帧或状态绘图）。

## 后续操作
  - 训练结束后，可以通过加载保存的模型进行推理和测试。
  - 若需要继续训练（恢复训练），可以将配置文件中的 `resume` 参数设置为 `True` ，这样训练脚本会加载上一次保存的模型状态继续训练。

## 二开总结

- 如果你要二次开发,你想自己添加一个变量,你就可以在这里`_init_buffers`加
- 一般来说我们做有地形的训练,有这种楼梯啊,嗯上下坡都是在 `trimesh` 里,然后如果你在平地平的话,你用 `plane` 就可以了
- `config`中的 `urdf` 文件位置需要修改
  - 青龙：`keypoints`：躯干 `link`
  - `foot_name`：脚的`link`
  - `stiffness` 和 `damping` 是关节的阻尼系数和刚度系数，对应 `urdf` 文件中的的名称
    - 给的都是 `revolute joint`，`fixed joint` 可以不用给，如果不是 `fixed` 需要给
- `envs` 下的 `__init__.py` 中注册环境，可以在这里添加自己的环境
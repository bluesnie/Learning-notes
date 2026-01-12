###### datetime:2025/12/28 12:25

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# View
## 加载模型

`mujoco.MjModel.from_xml_path`

## 数据结构

- `mjModel`，用来存储模型文件信息
- `mjData`，用来存储仿真数据

## 仿真运行接口

**mj_step**        

- 前向动力学（计算加速度 `qacc`）
- 数值积分（更新状态 `qpos`, `qvel`）
- 处理碰撞和约束
- 更新传感器数据

**mj_step1**       

- 计算当前状态下的广义加速度 (`qacc`)
- 处理碰撞检测，但不应用约束

**mj_step2**       

- 应用用户设置的 `ctrl` 和 `xfrc_applied`
- 处理约束（接触力、关节限位等）
- 数值积分更新状态 (`qpos`, `qvel`)
- 更新仿真时间 `d->time`

**mj_forward**     

前向动力学，但不推进仿真时间
给定关节位置(`qpos`)、速度(`qvel`)和关节力矩(`τ`)，计算关节加速度(`qacc`)

**mj_inverse**

给定关节位置(`qpos`)、速度(`qvel`)、加速度(`qacc`)，计算所需的关节力矩(`τ`)


```python
import time
import math

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../API-MJCF/pointer.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  cnt = 0
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    '''测试step '''
    # 这是一个数组，表示执行器的控制信号。在MuJoCo中，执行器可以是电机、位置伺服等。给d.ctrl赋值就是设置执行器的控制输入。例如，d.ctrl[0] = math.sin(cnt)就是让第一个执行器按照正弦波变化。
    # d.ctrl[1] = math.sin(cnt)
    # 自动完成：前向动力学 → 积分 → 更新状态
    # mujoco.mj_step(m, d)
    
    '''测试step1 step2 '''
    # 可以在两个阶段之间修改数据,需要中间干预的高级控制
    # mujoco.mj_step1(m, d)
    # d.ctrl[1] = math.sin(cnt)
    # mujoco.mj_step2(m, d)
    
    '''测试forward '''
    # d.ctrl[0] = math.sin(cnt)
    # # d.qpos[0] = math.sin(cnt) # 直接设定模型位置,用于：重置状态、设置初始位置、强制位置,注意：这跳过了物理规律！
    # mujoco.mj_forward(m, d)
    # print("qvel:",d.qvel)
    # print("qacc:",d.qacc)
    # print("qpos:",d.qpos)
    
    '''测试inverse '''
    d.qacc[0] = math.sin(cnt)
    d.qpos[0] = 0
    d.qvel[0] = 0
    mujoco.mj_inverse(m, d)
    # d.qfrc_inverse - 产生指定加速度所需的力,已知运动求所需的控制力（逆向问题）
    print("qfrc_inverse",d.qfrc_inverse) 
    
    cnt += 0.01

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
```

```xml
<mujoco>
    <compiler angle="radian" meshdir="meshes" autolimits="true" />
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" density="1.225" viscosity="1.8e-5" />
    <asset>
        <mesh name="tetrahedron" vertex="0 0 0 1 0 0 0 1 0 0 0 1" />
        <texture type="skybox" file="../MJCF/asset/desert.png"
            gridsize="3 4" gridlayout=".U..LFRB.D.." />
        <texture name="plane" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".9 .9 .9"
            width="512" height="512" mark="cross" markrgb=".8 .8 .8" />
        <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true" />
    </asset>

    <worldbody>
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="plane"
            condim="3" />
        <light directional="true" ambient=".3 .3 .3" pos="30 30 30" dir="0 -.5 -1"
            diffuse=".5 .5 .5" specular=".5 .5 .5" />

        <body name="base" pos="0 0 0" >
            <geom type="cylinder" mass="100" size="0.05 0.5" rgba=".2 .2 .2 1" />
            <body name="pointer" pos="0 0 0.51">
                <joint type="hinge" name="pivot" pos="0 0 0" axis="0 0 1" damping="0"
                    frictionloss="0" stiffness="0" />
                <geom type="capsule" mass="0.01" fromto="0 0 0 0.2 0 0" size="0.01"
                    rgba="0.8 0.2 0.2 0.5" />
                    <site name="imu" pos="0.05 0 0.02" size="0.02" rgba="0 0 1 .5"/>
                    <camera name="this_camera" mode="fixed" pos="0.4 0 0.1" euler="0 1.2 1.57" principalpixel="50 50" focalpixel="1080 1920" sensorsize="4 4" resolution="1280 1080"/>
            </body>
        </body>

        <body>
            
        </body>
    </worldbody>

    <actuator>
        <motor name="motor" joint="pivot"/>
        <velocity name="vel" joint="pivot" kv="10"/>
    </actuator>

    <sensor>
        <framequat name='quat' objtype='site' objname='imu' />
        <gyro name='ang_vel' site='imu' />
        <accelerometer name="accel" site="imu" />
        <jointpos name='pivot_p' joint='pivot' />
        <jointvel name='pivot_v' joint='pivot' />
        <framelinvel name="linvel" objtype="site" objname="imu"/>
    </sensor>

</mujoco>
```
###### datetime:2025/12/28 12:25

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# keyframe

用来调整 mujoco 的初始状态

**name=""**        
**time=""**        
&emsp;&emsp;时间，用来储存时刻   
**qpos="nq"**        
&emsp;&emsp;关节位置   
**qvel="nq"**        
&emsp;&emsp;关节角速度   
**act="na"**        
&emsp;&emsp;执行器数据，比如扭矩或者力   
**ctrl="nu"**        
&emsp;&emsp;控制输入   
**mpos="real(3*mjModel.nmocap)"**        
&emsp;&emsp;动捕body的pos   
**mquat="real(4*mjModel.nmocap)"**        
&emsp;&emsp;动捕body的quat   
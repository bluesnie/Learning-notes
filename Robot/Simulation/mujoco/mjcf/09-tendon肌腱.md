###### datetime:2025/12/27 12:51

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# tendon 肌腱
&emsp;&emsp;肌腱的作用就是将关节组合映射，可以将多个关节组合成一个控制器进行控制。最简单的用法可以看官方模型的 car.xml。这个模型将左轮和右轮统一映射成了向前和旋转两个控制器对车辆进行控制。如果作为麦轮或者全向轮来说可以分成 x,y,roat这三个控制器。       
![](./imgs/tendon1.png)       
![](./imgs/tendon2.png)       
&emsp;&emsp;tendon有两种组合模式，一种是spatial一种是 fixed。       

## spatial 

&emsp;&emsp;使用tendon组合关节驱动的时候，要使用tendon下面的spatial节点。spatial是类似一种像肌肉一样的驱动方式，比如线驱灵巧手等。如下图所示的驱动方式，通过每个site来拉住关节驱动。        
![](./imgs/spatial.png)       

&emsp;&emsp;spatial包含name.class,group,limited,rgba        
**range="0 0"**     
&emsp;&emsp;肌腱长度范围        
**frictionloss="0"**        
&emsp;&emsp;摩擦损失        
**width="0.003"**       
&emsp;&emsp;肌腱半径，可视化部分           
**stiffness="0"**       
&emsp;&emsp;刚性系数，相当于弹簧的弹力系数      
**damping="0"**     
&emsp;&emsp;阻尼系数。正值会产生沿肌腱作用的阻尼力（速度线性）。与 通过欧拉方法隐式积分的关节阻尼，则肌腱阻尼不是隐式积分的，因此 如果可能，应使用关节阻尼）。      

## fixed 

&emsp;&emsp;使用tendon组合关节映射的时候，要使用tendon下面的fixed节点。fixed中包含关节的映射关系。fixed中joint为制定关节，coed为缩放系数。原理就是使用fixed组合之后，控制器不再使用关节控制，而是将数据传给tendon/fixed，通过code缩放参数后给joint。    

## 肌肉

```xml
    <actuator>
        <muscle name="A" tendon="A" ctrlrange="-15 15"/>
    </actuator>
```

- 麦克纳姆轮

```xml
<mujoco model="example">
    <compiler angle="radian" meshdir="meshes" autolimits="true" />
    <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast" />
    <asset>
        <texture type="skybox" file="./imgs/desert.png"
            gridsize="3 4" gridlayout=".U..LFRB.D.." />
        <texture name="plane" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".9 .9 .9"
            width="512" height="512" mark="cross" markrgb=".8 .8 .8" />
        <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true" />
    </asset>
    <visual>
        <!-- 质量 -->
        <quality shadowsize="16384" numslices="28" offsamples="4" />
        <headlight diffuse="1 1 1" specular="0.5 0.5 0.5" active="0" />
    </visual>

    <worldbody>
        <light directional="true" pos="0 0 30" dir="0 -0.5 -1" />
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="plane"
            condim="3" />
        <!-- <geom type="box" size="0.3 0.3 0.5"/> -->

        <body name="base_body" pos="0 0 1">
            <freejoint />
            <geom type="box" size="0.5 0.5 0.01" rgba="0.5 0.5 0.5 1" mass="15" />

            <body name="Mecanum_A1" pos="0.5 0.6 0" euler="1.5707963267948966 0 0">
                <joint name="a1" type="hinge" pos="0 0 0" axis="0 0 1" frictionloss=".000002" />
                <geom type="cylinder" size="0.05 0.1" rgba=".1 .1 .1 1" />
                <site type="cylinder" size="0.15 0.045" rgba=".5 .1 .1 .5" />
                <replicate count="16" euler="0 0 0.3925">
                    <body euler="-0.78539815 0 0">
                        <joint type="hinge" pos="0.15 0 0" axis="0 0 1" frictionloss=".000002" />
                        <geom type="capsule" size="0.015 0.05" pos="0.15 0 0" />
                    </body>
                </replicate>
            </body>

            <body name="Mecanum_B1" pos="-0.5 0.6 0" euler="1.5707963267948966 0 0">
                <joint name="b1" type="hinge" pos="0 0 0" axis="0 0 1" frictionloss=".000002" />
                <geom type="cylinder" size="0.05 0.1" rgba=".1 .1 .1 1" />
                <site type="cylinder" size="0.15 0.045" rgba=".1 .1 .5 .5" />
                <replicate count="16" euler="0 0 0.3925">
                    <body euler="0.78539815 0 0">
                        <joint type="hinge" pos="0.15 0 0" axis="0 0 1" frictionloss=".000002" />
                        <geom type="capsule" size="0.015 0.05" pos="0.15 0 0" />
                    </body>
                </replicate>
            </body>

            <body name="Mecanum_A2" pos="-0.5 -0.6 0" euler="1.5707963267948966 0 0">
                <joint name="a2" type="hinge" pos="0 0 0" axis="0 0 1" frictionloss=".000002" />
                <geom type="cylinder" size="0.05 0.1" rgba=".1 .1 .1 1" />
                <site type="cylinder" size="0.15 0.045" rgba=".5 .1 .1 .5" />
                <replicate count="16" euler="0 0 0.3925">
                    <body euler="-0.78539815 0 0">
                        <joint type="hinge" pos="0.15 0 0" axis="0 0 1" frictionloss=".000002" />
                        <geom type="capsule" size="0.015 0.05" pos="0.15 0 0" />
                    </body>
                </replicate>
            </body>

            <body name="Mecanum_B2" pos="0.5 -0.6 0" euler="1.5707963267948966 0 0">
                <joint name="b2" type="hinge" pos="0 0 0" axis="0 0 1" frictionloss=".000002" />
                <geom type="cylinder" size="0.05 0.1" rgba=".1 .1 .1 1" />
                <site type="cylinder" size="0.15 0.045" rgba=".1 .1 .5 .5" />
                <replicate count="16" euler="0 0 0.3925">
                    <body euler="0.78539815 0 0">
                        <joint type="hinge" pos="0.15 0 0" axis="0 0 1" frictionloss=".000002" />
                        <geom type="capsule" size="0.015 0.05" pos="0.15 0 0" />
                    </body>
                </replicate>
            </body>
        </body>

    </worldbody>
    <tendon>
        <fixed name="forward">
            <joint joint="a1" coef="0.25" />
            <joint joint="b1" coef="0.25" />
            <joint joint="a2" coef="0.25" />
            <joint joint="b2" coef="0.25" />
        </fixed>
        <fixed name="transverse" frictionloss="0.001">
            <joint joint="a1" coef=".25" />
            <joint joint="b1" coef="-0.25" />
            <joint joint="a2" coef=".25" />
            <joint joint="b2" coef="-0.25" />
        </fixed>
        <fixed name="roatate">
            <joint joint="a1" coef=".25" />
            <joint joint="b1" coef=".25" />
            <joint joint="a2" coef="-.25" />
            <joint joint="b2" coef="-.25" />
        </fixed>
    </tendon>
    <actuator>
        <velocity tendon="forward" name="forward" kv="30" ctrlrange="-15 15"/>
        <velocity tendon="transverse" name="transverse" kv="30" ctrlrange="-15 15"/>
        <velocity tendon="roatate" name="roatate" kv="30" ctrlrange="-15 15"/>
    </actuator>
</mujoco>
```

- 全向轮

```xml
<mujoco model="example">
    <compiler angle="radian" meshdir="meshes" autolimits="true" />
    <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast" />
    <asset>
        <texture type="skybox" file="./imgs/desert.png"
            gridsize="3 4" gridlayout=".U..LFRB.D.." />
        <texture name="plane" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".9 .9 .9"
            width="512" height="512" mark="cross" markrgb=".8 .8 .8" />
        <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true" />
    </asset>
    <visual>
        <!-- 质量 -->
        <quality shadowsize="16384" numslices="28" offsamples="4" />
        <headlight diffuse="1 1 1" specular="0.5 0.5 0.5" active="0" />
    </visual>

    <worldbody>
        <light directional="true" pos="0 0 30" dir="0 -0.5 -1" />
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="plane"
            condim="3" />

        <body name="base_body" pos="0 0 1">
            <freejoint />
            <geom type="cylinder" size="0.5 0.01" rgba=".6 .6 .6 .5" />
            <replicate count="4" euler="0 0 1.57">
                <body name="om_m" pos="0.6 0 0" quat="0.707107 0 0.707107 0">
                    <joint name="a_m" type="hinge" axis="0 0 1" frictionloss=".002" />
                    <geom type="cylinder" size="0.05 0.1" rgba=".1 .1 .1 1" />
                    <geom type="cylinder" size="0.15 0.045" rgba=".1 .1 .5 .5" />
                    <body name="mini_wheel1" pos="0 0 0.015" euler="0 0 0.224285714">
                        <replicate count="14" euler="0 0 0.448571429">
                            <body euler="1.5707963267948966 0 0">
                                <joint type="hinge" pos="0.15 0 0" axis="0 0 1"
                                    frictionloss=".00002" />
                                <geom type="capsule" size="0.01 0.02" pos="0.15 0 0" />
                            </body>
                        </replicate>
                    </body>
                    <body name="mini_wheel2" pos="0 0 -0.015" euler="0 0 0">
                        <replicate count="14" euler="0 0 0.448571429">
                            <body euler="1.5707963267948966 0 0">
                                <joint type="hinge" pos="0.15 0 0" axis="0 0 1"
                                    frictionloss=".00002" />
                                <geom type="capsule" size="0.01 0.02" pos="0.15 0 0" />
                            </body>
                        </replicate>
                    </body>
                </body>
            </replicate>
        </body>

    </worldbody>

    <tendon>
        <fixed name="forward" frictionloss="0.001">
            <joint joint="a_m0" coef=".25" />
            <joint joint="a_m2" coef="-.25" />
        </fixed>
        <fixed name="transverse" >
            <joint joint="a_m1" coef=".25" />
            <joint joint="a_m3" coef="-.25" />  
        </fixed>
        <fixed name="roatate">
            <joint joint="a_m0" coef=".25" />
            <joint joint="a_m1" coef=".25" />
            <joint joint="a_m2" coef=".25" />
            <joint joint="a_m3" coef=".25" />  
        </fixed>
    </tendon>    


    <actuator>
        <velocity tendon="forward" name="forward" kv="30" ctrlrange="-15 15"/>
        <velocity tendon="transverse" name="transverse" kv="30" ctrlrange="-15 15"/>
        <velocity tendon="roatate" name="roatate" kv="30" ctrlrange="-15 15"/>
    </actuator>
    
</mujoco>
```
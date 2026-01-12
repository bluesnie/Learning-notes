###### datetime:2025/12/28 12:25

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# default

默认参数，可以赋给实际应用的各种对应实体属性参数，使用 `class`指定，类似于 `CSS`中的 `class`。

```xml
<mujoco>
    <asset>
        <mesh name="tetrahedron" vertex="0 0 0 1 0 0 0 1 0 0 0 1" />
        <texture type="skybox" file="../asset/desert.png"
            gridsize="3 4" gridlayout=".U..LFRB.D.." />
        <texture name="plane" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".9 .9 .9"
            width="512" height="512" mark="cross" markrgb=".8 .8 .8" />
        <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true" />
        <texture name="y2c" type="cube" builtin="gradient" rgb1="1 1 0" rgb2="0 1 1"
            width="512" height="512" />
        <material name="y2c" reflectance="0.3" texture="y2c" texrepeat="1 1" texuniform="true" />
        <texture name="r2b" type="cube" builtin="gradient" rgb1="1 0 0" rgb2="0 0 1"
            width="512" height="512" />
        <material name="r2b" reflectance="0.3" texture="r2b" texrepeat="1 1" texuniform="true" />
    </asset>
    <default>
        <default class="collision_box">
            <geom type="box" size="0.1 0.1 0.1" contype="1" conaffinity="1" material="y2c" />
        </default>
        <default class="visual_box">
            <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0" material="r2b" />
        </default>
        <default class="collision">
            <geom contype="1" conaffinity="1" material="y2c" />
        </default>
        <default class="visual">
            <geom contype="0" conaffinity="0" material="r2b" />
        </default>
    </default>

    <worldbody>
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="plane"
            condim="3" />
        <light directional="true" ambient=".3 .3 .3" pos="30 30 30" dir="0 -.5 -1"
            diffuse=".5 .5 .5" specular=".5 .5 .5" />

        <body pos="0 0 1">
            <freejoint/>
            <geom pos="0 0 0.2" class="visual_box" />
            <geom class="collision_box" />
        </body>

        <body pos="0 1 1">
            <freejoint/>
            <geom type="sphere" size=".1"  pos="0 0 0.2" class="visual" />
            <geom type="cylinder" size=".1 .1 .1" class="collision" />
        </body>

    </worldbody>

</mujoco>
```


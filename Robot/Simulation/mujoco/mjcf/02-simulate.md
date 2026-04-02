###### datetime:2025/12/27 12:51

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# simulate

## 启动
编译和release版本的simulate程序都在bin路径下面
python版本可使用python -m mujoco.viewer启动
## 常用操作
* 双击选中的物体并高亮
* 双击选中物体后，按住ctrl+左键是调整姿态
* 双击选中物体后，按住ctrl+右键是在双击选中的位置施加一个力
* +和-是仿真世界时间流逝速度
* ctrl+A相机视角回正
* 空格键暂停恢复
## 快捷键

|key|function|
|---|---|
|F1|help|
|F2|info|
|F3|profiler|
|F4|sensors|
|F5|全屏|
|F6|切换可视化坐标系-对于左侧`Rendering`的`frame`坐标系|
|F7|实体名字标签-对于左侧`Rendering`的`Label`|
|TAB|隐藏左侧工具栏|
|Delete|重置世界|
|`|geom最小外界矩形和碰撞状态|
|Q|Camera可视化，方括号切换相机视角|
|W|世界网格化|
|E|Equality|
|R|开关光线反射|
|T|几何体透明化|
|Y|测距可视化|
|U|驱动器方向可视化|
|I|转动惯量可视化|
|O|调整物体位姿可视化|
|P|Contact可视化|
|[ / ]|切换相机视角|
| \ |Mesh Tree|
|A|auto Connect|
|D|只显示body|
|S|画面亮度调整|
|F|接触力大小及方向可视化，一般配合`C`接触点使用|
|G|迷雾|
|J|关节方向可视化|
|H|凸包可视化|
|J|关节选装方向可视化|
|K|关闭天空盒|
|;|Skin可视化|
|'|缩放转动惯量|
|Z|灯光|
|X|Texture关闭|
|C|接触点可视化|
|V|肌腱可视化|
|B|扰动力大小及方向可视化|
|M|质心可视化|
|,|Activation|
|/|haze 地平线|
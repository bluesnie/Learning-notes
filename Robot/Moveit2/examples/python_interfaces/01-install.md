###### datetime:2025/02/17 16:24

###### author:nzb

## 安装

- [main分支安装](https://moveit.ai/install-moveit2/source/)
  - `main` 分支才有`moveit_py`包
- 编译`colcon build --event-handlers desktop_notification- status- --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip pilz_industrial_motion_planner`
  - 跳过`pilz_industrial_motion_planner`规划器，`Eigen`相关报错，跳过需要修改`motion_planning_python_api_tutorial.yaml`文件，将`pilz_industrial_motion_planner`注释掉

## 问题及解决方案

### 问题1：

```shell
Finished <<< geometric_shapes [39.1s]
Finished <<< moveit_msgs [42.8s]
Starting >>> moveit_core
--- stderr: moveit_core
In file included from /home/robolab/ws_moveit2/src/moveit2/moveit_core/online_signal_smoothing/src/butterworth_filter.cpp:39:
/home/robolab/ws_moveit2/src/moveit2/moveit_core/online_signal_smoothing/include/moveit/online_signal_smoothing/butterworth_filter.hpp:46:10: fatal error: moveit_core/moveit_butterworth_filter_parameters.hpp: No such file or directory
   46 | #include <moveit_core/moveit_butterworth_filter_parameters.hpp>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
gmake[2]: *** [online_signal_smoothing/CMakeFiles/moveit_butterworth_filter.dir/build.make:76: online_signal_smoothing/CMakeFiles/moveit_butterworth_filter.dir/src/butterworth_filter.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:1828: online_signal_smoothing/CMakeFiles/moveit_butterworth_filter.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
gmake: *** [Makefile:146: all] Error 2
---
Failed   <<< moveit_core [19.2s, exited with code 2]

Summary: 13 packages finished [1min 2s]
  1 package failed: moveit_core
  2 packages had stderr output: geometric_shapes moveit_core
  37 packages not processed
```

- 解决方案：`sudo apt install ros-humble-generate-parameter-library*`

### 问题2：

- 存在 `osqp` 找不到问题
- 解决方案

```shell
git clone https://github.com/osqp/osqp.git
cd osqp && mkdir build && cd build
cmake .. && make && sudo make install
```

### 问题3：

- `google-benchmark` 找不到问题
- 解决方法：`apt-get install ros-humble-ament-cmake-google-benchmark`
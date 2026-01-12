###### datetime:2025/12/28 12:25

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# get obj

## 获取名字
数量
![](../../mjcf/imgs/entity.png)
**先看mjModel结构体开头部分，这里的命名方式都是nXXX，这代表各个元素的数量**
![](../../mjcf/imgs/names.png)

## 获取id
`MJAPI int mj_name2id(const mjModel* m, int type, const char* name);`
通过 `name` 获取实体的id 
- `m:mjModel`
- `type`: `mjmodel.h`文件中的`mjtObj`中定义，这个是要获取`id`的实体类型一下是部分`type`类型枚举，在`mjtObj`中找到       
- `name: name`

![](../../mjcf/imgs/enum_mjtobj.png)

## 获取位置
![](../../mjcf/imgs/xpos.png)
**可以通过 xpos和xxx_xpos获取各个对象的位置**


## 代码

- `get_obj.cc`

```cpp
// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mujoco/mjmodel.h>
#include <thread>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods) {
  // update button state
  button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// main function
int main(int argc, const char **argv) {

  char error[1000] = "Could not load binary model";
  m = mj_loadXML("../../../../API-MJCF/pointer.xml", 0, error, 1000);

  // make data
  d = mj_makeData(m);

  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  GLFWwindow *window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  //获取名字
  auto print = [&](int num, int *adr) {
    std::cout << "num:" << num << std::endl;
    std::cout << "name:";
    for (int i = 0; i < num; i++)
      std::cout << "  " << m->names + adr[i];
    std::cout << std::endl;
  };
//   print(m->njnt, m->name_jntadr);
  print(m->nbody, m->name_bodyadr);

  //获取id
  int pointer_id = mj_name2id(m, mjOBJ_BODY, "pointer");
  std::cout << "pointer id:" << pointer_id << std::endl;
  int imu_id = mj_name2id(m, mjOBJ_SITE, "imu");
  std::cout << "imu id:" << imu_id << std::endl;

  auto step_start = std::chrono::high_resolution_clock::now();
  while (!glfwWindowShouldClose(window)) {

    mj_step(m, d);

    //获取位置
    // body
    mjtNum *pos = d->xpos + pointer_id * 3; // 因为 xpos 是3个数据，所以 要乘以3
    std::cout << "pointer pos:" << pos[0] << "  " << pos[1] << "  " << pos[2]
              << std::endl;
    mjtNum *imu_pos = d->site_xpos + imu_id * 3;
    std::cout << "imu pos:" << imu_pos[0] << "  " << imu_pos[1] << "  "
              << imu_pos[2] << std::endl;

    // //获取姿态 w x y z
    mjtNum *quat = d->xquat + pointer_id * 4; // 因为 xquat 是4个数据，所以 要乘以4
    std::cout << "quat:" << quat[0] << "  " << quat[1] << "  " << quat[2]
              << "  " << quat[3] << std::endl;

    //同步时间
    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec =
        std::chrono::duration<double>(current_time - step_start).count();
    double time_until_next_step = m->opt.timestep * 5 - elapsed_sec;
    if (time_until_next_step > 0.0) {
      auto sleep_duration = std::chrono::duration<double>(time_until_next_step);
      std::this_thread::sleep_for(sleep_duration);
    }

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  // free visualization storage
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // free MuJoCo model and data
  mj_deleteData(d);
  mj_deleteModel(m);

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 1;
}
```

- `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(MUJOCO_T)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)

#编译安装，从cmake安装位置opt使用

# 设置 MuJoCo 的路径
set(MUJOCO_PATH "/home/nzb/programs/mujoco-3.3.0")
# 包含 MuJoCo 的头文件
include_directories(${MUJOCO_PATH}/include)
# 设置 MuJoCo 的库路径
link_directories(${MUJOCO_PATH}/bin)
set(MUJOCO_LIB ${MUJOCO_PATH}/lib/libmujoco.so)


add_executable(get_obj get_obj.cc )
#从cmake安装位置opt使用
target_link_libraries(get_obj ${MUJOCO_LIB} glut GL GLU glfw)
```

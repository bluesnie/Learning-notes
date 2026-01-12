###### datetime:2025/12/28 12:25

###### author:nzb

> 该项目来源于[mujoco_learning](https://github.com/Albusgive/mujoco_learning)

# 编译安装版使用 mujoco

## cmake install到opt中
可以直接使用cmake的find_package寻找到mujoco库
```CMake
set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
if (mujoco_FOUND)
message(STATUS "Find mujoco")
endif()
```
然后是链接库
`target_link_libraries(your_app mujoco::mujoco glut GL GLU glfw)`

## cmake不安装

编译后没有使用cmake安装也可以直接在编译出寻找mujoco库
```CMake
set(MUJOCO_PATH "/home/albusgive/software/mujoco-3.3.1")
include_directories(${MUJOCO_PATH}/include)
link_directories(${MUJOCO_PATH}/build/bin)
set(MUJOCO_LIB ${MUJOCO_PATH}/build/lib/libmujoco.so)
```
链接库：
`target_link_libraries(your_app ${MUJOCO_LIB} glut GL GLU glfw)`

# Release安装版使用 mujoco

和第二类只有路径有区别
```CMake
# 设置 MuJoCo 的路径
set(MUJOCO_PATH "/home/albusgive/software/mujoco")
# 包含 MuJoCo 的头文件
include_directories(${MUJOCO_PATH}/include)
# 设置 MuJoCo 的库路径
link_directories(${MUJOCO_PATH}/bin)
set(MUJOCO_LIB ${MUJOCO_PATH}/lib/libmujoco.so)
```
链接库：
`target_link_libraries(your_app ${MUJOCO_LIB} glut GL GLU glfw)`

# 测试环境

<font color=Red >注意：编译的时候可能缺GLFW 使用 sudo apt-get install libglfw3-dev </font>      
把官方提供的sample中的basic.cc复制过来      
编译：      
```
mkdir build
cd build
cmake ..
make
./basic ../../../API-MJCF/pointer.xml
```

# 编译simulate测试

- 1 .将 mujoco文件夹中的 simulate的文件夹复制出来，只保留以下文件：
![](../../MJCF/asset/simulate_src.png)
- 2.liblodepng.a在编译好的mujoco/build/lib中,lodepng.h在mujoco/build/_deps/lodepng-src中,拷到你的simulate中
- 3 .CMakeLists.txt如下:

```CMake
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)
file(GLOB SIMULATE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/simulate/*.cc)
add_executable(simulate ${SIMULATE_SRC})
target_link_libraries(simulate mujoco::mujoco glut GL GLU glfw lodepng)
```

- 4 .编译运行
```
cmake ..
make
./simulate
```




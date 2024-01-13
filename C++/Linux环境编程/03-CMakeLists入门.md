###### datetime:2023/04/12 14:45

###### author:nzb

# CMakeLists入门

makefile文件的编写实在是个繁琐的事，于是，CMake出现了，使得这一切变得简单，CMake通过CMakeLists.txt读入所有源文件自动生成makefile，进而将源文件编译成可执行文件或库文件

## 一、CMake常用的命令

```makefile
# 设置cmake最低版本
cmake_minimum_required(VERSION 3.2)

# project命令用于指定cmake工程的名称，实际上，它还可以指定cmake工程的版本号（VERSION关键字）、简短的描述（DESCRIPTION关键字）、主页URL（HOMEPAGE_URL关键字）和编译工程使用的语言（LANGUAGES关键字）
# project(<PROJECT-NAME> [<language-name>...])
# project(<PROJECT-NAME> [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]] [DESCRIPTION <project-description-string>][HOMEPAGE_URL <url-string>] [LANGUAGES <language-name>...])
# ${PROJECT_SOURCE_DIR} 和 <PROJECT-NAME>_SOURCE_DIR：本CMakeLists.txt所在的文件夹路径
# ${PROJECT_NAME}：本CMakeLists.txt的project名称
project(xxx)
project(mytest VERSION 1.2.3.4)
project (mytest HOMEPAGE_URL “https://www.XXX(示例).com”)

# 获取路径下所有的.cpp/.c/.cc文件（不包括子目录），并赋值给变量中
aux_source_directory(路径 变量)

# GLOB 获取目录下的所有cpp文件（不包括子目录），并赋值给SOURCES
file(
        GLOB SOURCES
        ${PROJECT_SOURCE_DIR}/*.c

)
# GLOB_RECURSE 获取目录下的所有cpp文件（包括子目录），并赋值给NATIVE_SRC
file(
      GLOB_RECURSE NATIVE_SRC 
      ${PROJECT_SOURCE_DIR}/lib/*.cpp
)

# 给文件名/路径名或其他字符串起别名，用${变量}获取变量内容
set(变量 文件名/路径/...)

# 添加编译选项FOO BAR
# add_definitions定义宏，但是这种定义方式无法给宏具体值 等价C语言中的#define  MG_ENABLE_OPENSSL
add_definitions(-DFOO -DBAR ...)

# add_compile_definitions定义宏，这种方式可以给宏具体值，但是这个指令只要高版本的cmake支持 等价C语言中 #define  MG_ENABLE_OPENSSL   1 
add_compile_definitions(MG_ENABLE_OPENSSL=1)

# 打印消息
message(消息)

# 编译子文件夹的CMakeLists.txt
add_subdirectory(子文件夹名称)

# 将.cpp/.c/.cc文件生成.a静态库
# 注意，库文件名称通常为libxxx.so，在这里只要写xxx即可
add_library(库文件名称 STATIC 文件)

# 将.cpp/.c/.cc文件生成可执行文件
add_executable(可执行文件名称 文件)

# 规定.h头文件路径
include_directories(路径)

# 规定.so/.a库文件路径
link_directories(路径)

# 设置编译选项及默认值
option(TEST_DEBUG "option for debug" OFF)

# 对add_library或add_executable生成的文件进行链接操作
# 注意，库文件名称通常为libxxx.so，在这里只要写xxx即可
target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)
```

通常一个CMakeLists.txt需按照下面的流程：

```text
project(xxx)                                          #必须

add_subdirectory(子文件夹名称)                         #父目录必须，子目录不必

add_library(库文件名称 STATIC 文件)                    #通常子目录(二选一)
add_executable(可执行文件名称 文件)                     #通常父目录(二选一)

include_directories(路径)                              #必须
link_directories(路径)                                 #必须

target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)       #必须
```

## 二、CMakeLists实例

### 示例1：只有一个源文件main.c

目录结构如下：

```text
+
| 
+--- main.c
+--- CMakeLists.txt
|
```

代码如下：

```c++
// main.c
#include <stdio.h>
int main()
{
printf("hello world");
return 0;
}
```

```text
# CMakeLists
cmake_minimum_required(VERSION 3.0)
project(HELLO VERSION 1.0 LANGUAGES C CXX)
set(SOURCES main.c)
add_executable(hello ${SOURCES})
```

⚠️警告：project设置VERSION,要求cmake的最低版本3.0

**注意：**

为了简单起见，我们从一开始就采用`cmake`的 `out-of-source` 方式来构建（即生成中间产物与源代码分离），并始终坚持这种方法，这也就是此处为什么单独创建一个目录，然后在该目录下执行 `cmake` 的原因

在CMakeLists.txt目录下执行以下命令

```text
mkdir build
cd build
cmake ..
make
```

即可生成可执行程序 `hello(.exe)`  
目录结构如下

```text
+
| 
+--- main.c
+--- CMakeLists.txt
|
/--+ build/
   |
   +--- hello(exec)
```

- `project` 会引入两个变量`HELLO_BINARY_DIR` 和 `HELLO_SOURCE_DIR`，这两个变量和`PROJECT_BINARY_DIR` 和 `PROJECT_SOURCE_DIR`等价
- `message(${PROJECT_SOURCE_DIR}) `打印变量的值
- `set` 命令用来设置变量
- `add_exectuable` 告诉工程生成一个可执行文件。
- `add_library` 则告诉生成一个库文件

### 示例2：拆成3个文件 hello.h hello.c main.c

目录结构如下：

```text
+
| 
+--- main.c
+--- hello.h
+--- hello.c
+--- CMakeLists.txt
|
```

代码如下：

```text
// main.c
#include "hello.h"
int main()
{
    hello("World");
    return 0;
}
```

```text
// hello.h
#ifndef DBZHANG_HELLO_
#define DBZHANG_HELLO_
void hello(const char* name);
#endif //DBZHANG_HELLO_
```

```text
// hello.c
#include <stdio.h>
#include "hello.h"

void hello(const char * name)
{
    printf ("Hello %s! \n", name);
}
```

```text
# CMakeLists
cmake_minimum_required(VERSION 3.0)
project(HELLO VERSION 1.0 LANGUAGES C CXX)
set(SOURCES hello.c main.c)
add_executable(hello ${SOURCES})
```

执行cmake的过程同上，目录结构

```text
+
| 
+--- main.c
+--- hello.h
+--- hello.c
+--- CMakeLists.txt
|
/--+ build/
   |
   +--- hello(exec)
```

### 示例3：在示例2的基础上，先将 hello.c 生成一个库hellolib，再给main.c使用

我们只需修改下CMakeLists即可

```text
# CMakeLists
cmake_minimum_required(VERSION 3.0)
project(HELLO VERSION 1.0 LANGUAGES C CXX)
set(LIB_SRC hello.c)
add_library(libhello ${LIB_SRC})
set(APP_SRC main.c)
add_executable(hello ${APP_SRC})
target_link_libraries(hello libhello)
```

执行cmake的过程同上，目录结构如下

```text
+
| 
+--- main.c
+--- hello.h
+--- hello.c
+--- CMakeLists.txt
|
/--+ build/
   |
   +--- liblibhello.a
   +--- hello(exec)
```

- `target_link_libraries` 该指令的作用为将目标文件与库文件进行链接。该指令的语法如下：
  `target_link_libraries(<target> [item1] [item2] [...] [[debug|optimized|general] <item>] ...)`
  上述指令中的`<target>`是指通过`add_executable()`和`add_library()`指令生成已经创建的目标文件。 而`[item]`
  表示库文件没有后缀的名字。默认情况下，库依赖项是传递的。当这个目标链接到另一个目标时，链接到这个目标的库也会出现在另一个目标的连接线上
  `target_link_libraries`里库文件的顺序符合gcc链接顺序的规则，即被依赖的库放在依赖它的库的后面，比如
  `target_link_libraries(hello A B.a C.so)`
  在上面的命令中，`libA.so`可能依赖于`libB.a`和`libC.so`，如果顺序有错，链接时会报错。还有一点，`B.a`会告诉`CMake`优先使用静态链接库`libB.a`，
  `C.so`会告诉CMake优先使用动态链接库`libC.so`，也可直接使用库文件的相对路径或绝对路径。使用绝对路径的好处 于，当依赖的库被更新时，make的时候也会重新链接

- `set_target_properties（...）`是一个便捷功能
    - 设置多个目标的多个属性：`set_target_properties(libhello PROPERTIES OUTPUT_NAME "hello")`
    - 重命名libhello为hello：
      `set_target_properties(Thirdlib PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/libThirdlib.so)`
    - cmakelist 添加依赖库：
    ```text
    set_target_properties(test PROPERTIES LINKER_LANGUAGE CXX) // 指定C++
    set_target_properties(test PROPERTIES LINKER_LANGUAGE C) // 指定C
    ```

该目标属性用于指定编译器的语言。即当调用可执行程序、共享库和模块时，用于指定编译器链接语言（C or CXX），若是没有设置，则默认具有最高链接器首选项值的语言

### 示例4：将源文件放置到不同的目录

在示例2的基础上，我们修改下目录结构

目录结构如下：

```text
+
|
+--- CMakeLists.txt
/--+ src/
|  |
|  +--- main.c
|  +--- CMakeLists.txt
|
/--+ libhello/
|  |
|  +--- hello.h
|  +--- hello.c
|  +--- CMakeLists.txt
```

顶层的CMakeLists.txt

```text
# CMakeLists
cmake_minimum_required(VERSION 3.0)
project(HELLO VERSION 1.0 LANGUAGES C CXX)
add_subdirectory(src)
add_subdirectory(libhello)
```

src的CMakeLists.txt

```text
# src CMakeLists
include_directories(${PROJECT_SOURCE_DIR}/libhello)
set(APP_SRC main.c)
add_executable(hello ${APP_SRC})
target_link_libraries(hello libhello)
```

libhello的CMakeLists.txt

```text
#libhello CMakeLists
set(LIB_SRC hello.c)
add_library(libhello ${LIB_SRC})
```

执行cmake的过程同上，目录结构如下

```text
+
|
+--- CMakeLists.txt
/--+ src/
   |
   +--- main.c
   +--- CMakeLists.txt

/--+ libhello/
   |
   +--- hello.h
   +--- hello.c
   +--- CMakeLists.txt

/--+ build/
   |
   / --+ src/
      | 
      +--- hello(exec)

   / --+ libhello/ 
      |
      +--- liblibhello.a
```

- `add_subdirectory (source_dir [binary_dir] [EXCLUDE_FROM_ALL])`
添加一个子目录并构建该子目录
  - `source_dir`：必选参数。该参数指定一个子目录，子目录下应该包含CMakeLists.txt文件和代码文件。子目录可以是相对路径也可以是绝对路径，如果是相对路径，则是相对当前目录的一个相对路径。
  - `binary_dir`：可选参数。该参数指定一个目录，用于存放输出文件。可以是相对路径也可以是绝对路径，如果是相对路径，则是相对当前输出目录的一个相对路径。如果该参数没有指定，则默认的输出目录使用source_dir。
  - `EXCLUDE_FROM_ALL`：可选参数。当指定了该参数，则子目录下的目标不会被父目录下的目标文件包含进去，父目录的CMakeLists.txt不会构建子目录的目标文件，必须在子目录下显式去构建。例外情况：当父目录的目标依赖于子目录的目标，则子目录的目标仍然会被构建出来以满足依赖关系（例如使用了target_link_libraries）
- `include_directories ([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])`
将指定目录添加到编译器的头文件搜索路径之下，指定的目录被解释成当前源码路径的相对路径

### 示例5：在示例4的基础上，将可执行文件和lib都放到对应的bin目录和lib目录下

方法一：修改顶层CMakeLists.txt中的add_subdirectory方法

```text
# 顶层CMakeLists
cmake_minimum_required(VERSION 3.0)
project(HELLO VERSION 1.0 LANGUAGES C CXX)
add_subdirectory(src ./bin)
add_subdirectory(libhello ./lib)
```

生成的可执行文件在build/bin中，生成的lib文件在build/lib中

方法二：修改其他两个文件CMakeLists.txt

```text
# src CMakeLists
include_directories(${PROJECT_SOURCE_DIR}/libhello)
set(APP_SRC main.c)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(hello ${APP_SRC})
target_link_libraries(hello libhello)
```

```text
# libhello CMakeLists
set(LIB_SRC hello.c)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(libhello ${LIB_SRC})
```

- `set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)` 设置可执行文件输出路径
- `set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)` 设置lib库输出路径

### 示例6：在示例4的基础上，编译动态库

`add_library(libhello SHARED ${LIB_SRC})`

```text
# 顶层 CMakeLists
cmake_minimum_required(VERSION 3.0)
project(HELLO VERSION 1.0 LANGUAGES C CXX)
option(TEST_DEBUG "option for debug" OFF)
if (TEST_DEBUG)
add_definitions(-DTEST_DEBUG)
endif()
add_definitions(-DBUILD_SHARED)
add_subdirectory(src)
add_subdirectory(libhello)
```

```text
# src CMakeLists
include_directories(${PROJECT_SOURCE_DIR}/libhello)
set(APP_SRC main.c)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(hello ${APP_SRC})
target_link_libraries(hello libhello)
```

```text
# libhello CMakeLists
set(LIB_SRC hello.c)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

if(BUILD_SHARED)
  add_library(libhello SHARED ${LIB_SRC})
else()
  add_library(libhello STATIC ${LIB_SRC})
endif()
```

我们在main.c中使用CMakeLists中定义的宏TEST_DEBUG

```text
// main.c
#include "hello.h"
#include <stdio.h>
int main()
{
    hello("World");
#ifdef TEST_DEBUG
    printf ("DEBUG \n");
#endif
    return 0;
}
```

执行cmake

```text
#生成动态库
cmake .. -DBUILD_SHARED=1   
```

```text
# main.c中的“DEBUG“会打印
cmake .. -DTEST_DEBUG=ON 
```
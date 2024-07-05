###### datetime:2024/06/25 19:44

###### author:nzb

# Windows安装lcm

- [中文链接](https://blog.csdn.net/m0_57186135/article/details/136301473)
- [原文链接](https://github.com/AmyPhung/virtual-control-room/wiki/LCM-Setup-(Windows))

## 安装注意事项

- `python`的`debug dll`
  - `LCM_ENABLE_PYTHON`打开，需要`debug`模式，需要重新安装`python`，安装 `ros2` 时候的`choco`安装的没开`debug`模式，目录跟`choco`一样为`C:/Python38`
  - [相关链接](https://blog.csdn.net/weixin_43788499/article/details/84933210)
- 第一次打开`Visual Studio`时，生成前必须：`Set solution configuration to “Release”`

## 无法打开文件libglib-2.0.dll.a 
  - 解决：`In cmake/FindGLib2.cmake, replace`

```text
find_library(GLIB2_${VAR}_LIBRARY NAMES ${LIB}-2.0 ${LIB})


replace to
 

if(WIN32)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a")
        set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
endif()
find_library(GLIB2_${VAR}_LIBRARY NAMES ${LIB}-2.0 ${LIB})
```

## 找不到`DLL`

【Python3.8】ctypes加载dll: 报错 FileNotFoundError: Could not find module ‘xx.dll’ (or one of its dependencies).

- 原因  
    自`python3.8`开始，考虑到`dll`劫持等安全性问题，`python3.8`更改了`dll`的搜索加载机制，即只在指定可行的位置搜索加载`dll`，不再自动读取系统的环境变量`Path`了。
- [官方解释](https://docs.python.org/3/library/os.html#os.add_dll_directory)

- 解决方法
  - 方法1
    - 老老实实使用完整的绝对路径, 如果还不行，说明这个dll依赖了其他路径下的dll，这种情况使用方法2添加路径
        ```python
            import ctypes
            ctypes.CDLL("xxx/yyy/zzz.dll")  
            # or
            ctypes.WinDLL("xxx/yyy/zzz.dll") 
        ```

  - 方法2  
    - 添加dll及其所有依赖的搜索路径（绝对路径）再加载
      ```python
        import ctypes
        os.add_dll_directory("xxx/yyy")
        ctypes.CDLL("zzz.dll")
      ```
  - 方法3（推荐）  
    在加载时加上参数`winmode=0`，此参数为`py38`的参数，为了兼容`3.8`以下，`3.8`以前的没有这个参数。
    此方法能够使用相对路径 加载工程下的`dll` 或者 加载环境变量`Path`下的`dll`
    ```python
        import ctypes
        ctypes.CDLL("xxx/yyy/zzz.dll", winmode=0)  
        # or
        ctypes.WinDLL("xxx/yyy/zzz.dll", winmode=0) 
    ```
  - `lcm`的解决
    ```python
        import ctypes, sys
        sys.path.append("C:\Program Files (x86)\lcm\lib\site-packages")
        ctypes.WinDLL("C:\Program Files (x86)\lcm\bin\lcm.dll", winmode=0)
        import lcm # 可以导入
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")  # 报错，bind: No error，应该需要配置组播信息
    ```

## lcm初始化的参数

- [provider组播参数](https://lcm-proj.github.io/lcm/doxygen_output/c_cpp/html/group__LcmC__lcm__t.html)

## 组播相关
- 组播需要特定的[`IP`段](https://github.com/lcm-proj/lcm/issues/206)`uses a specific IP range. The suitable ranges are 224.0.0.0 through 239.255.255.255 for IP4.`
- [Windows组播设置](https://blog.csdn.net/xiongjia516/article/details/132364200)
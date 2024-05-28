###### datetime:2024/05/27 10:32

###### author:nzb


# vscode 插件

- **Python** 插件
    - `autopep8`：`Python`插件
    - `Pylance`：`Python`插件
    - `Python`：`Python`插件
    - `Python Debugger`：`Python`插件
- **AI**插件
    - `CodeGeeX`：`AI`代码生成器
- **通讯插件**
    - `Serial Monitor`：串口监视器
- **工具插件**
    - `Markdown All in One`：`Markdown`插件

# 配置

- `c_cpp_properties.json`

```text
{
  "configurations": [
    {
      "browse": {
        "databaseFilename": "${default}",
        "limitSymbolsToIncludedHeaders": false
      },
      "includePath": [
        "/opt/ros/humble/include/**",
        "/home/quinn/cyan_ws/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_tf_tools/include/**",
        "/home/quinn/cyan_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_ros_control/include/**",
        "/home/quinn/cyan_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_rviz/include/**",
        "/usr/include/**"
      ],
      "name": "ROS",
      "intelliSenseMode": "gcc-x64",
      "compilerPath": "/usr/bin/gcc",
      "cStandard": "gnu11",
      "cppStandard": "c++14"
    }
  ],
  "version": 4
}
```
- `settings.json`

```json
{
    "python.autoComplete.extraPaths": [
        "/home/quinn/cyan_ws/install/robotic_grasping/lib/python3.10/site-packages",
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages"
    ],
    "python.analysis.extraPaths": [
        "/home/quinn/cyan_ws/install/robotic_grasping/lib/python3.10/site-packages",
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages",
        "./interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules",
        "./interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_common_modules",
        
        "/home/blues/vscode_projects/cy1_ws/install/cy_gripper_interfaces/local/lib/python3.10/dist-packages"
    ],
    // black formatter配置
    "[python]": {
        // "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.defaultFormatter": "ms-python.autopep8",
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.formatOnSave": true,
        "autopep8.args": [
            // "--max-line-length=128"
        ]
    },
}
```

- `launch.json`

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "--log-level=DEBUG",
                "--log-file=/tmp/debugpy.log"
            ]
        }
    ]
}
```

# 快捷键设置

- 折叠展开
  - 折叠：左下角齿轮设置->`Keyboard Shortcuts`->`Fold all` -> `Ctrl + Alt + C`
  - 展开：左下角齿轮设置->`Keyboard Shortcuts`->`Unfold all` -> `Ctrl + Alt + O`
- 设置滚轮+ctrl放大和缩小字体
  - 点击左下角的设置按钮，弹出的菜单中点击`Settings`，搜索`Mouse Wheel Zoom`
  - 找到如图的`User`标签页下的`Text Editor`下的`Mouse Wheel Zoom`
  - 勾选选项，即可实现滚轮的放大和缩小


# 操作

- Python操作
  - 选择解释器
    - `Ctrl + Shift + p` -> `Python: Select Interperter` -> 选择对应的环境解释器














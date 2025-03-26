###### datetime:2025/03/26 14:41:00

###### author:nzb

# systemd 启动 python opencv 播放视频

## 常见报错

```text
Mar 26 10:35:31 pasture-10 systemd[1]: Started VLC media player..
Mar 26 10:35:31 pasture-10 python[21660]: qt.qpa.xcb: could not connect to display :0.0
Mar 26 10:35:31 pasture-10 python[21660]: qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/blues/anaconda3/envs/speech_interaction/lib/python3.8/site-packages/cv2/qt/plugins" even tho>
Mar 26 10:35:31 pasture-10 python[21660]: This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
Mar 26 10:35:31 pasture-10 python[21660]: Available platform plugins are: xcb.
```

## 初步尝试

要解决通过systemd服务启动时Qt平台插件加载失败的问题，请按照以下步骤操作：

1. **确认Qt平台插件路径**：
   在conda环境中，Qt平台插件通常位于`${CONDA_PREFIX}/plugins/platforms`。检查路径是否存在：
   ```bash
   ls /home/blues/anaconda3/envs/speech_interaction/plugins/platforms/libqxcb.so
   ```

2. **修改systemd服务文件**：
   添加必要的环境变量，特别是`QT_QPA_PLATFORM_PLUGIN_PATH`和`LD_LIBRARY_PATH`：

   ```ini
   [Unit]
   Description=Test Cv media player.
   After=graphical.target

   [Service]
   ExecStart=bash /home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/startup.sh
   # ExecStart=/home/blues/anaconda3/envs/speech_interaction/bin/python /home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/startup.py
   Restart=on-abort
   User=blues
   Group=blues
   Environment=DISPLAY=:0
   Environment=XAUTHORITY=/home/blues/.Xauthority
   #Environment=QT_DEBUG_PLUGINS=1
   Environment=QT_QPA_PLATFORM_PLUGIN_PATH=/home/blues/anaconda3/envs/speech_interaction/plugins/platforms
   Environment=LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/blues/anaconda3/envs/speech_interaction/lib

   [Install]
   # 图形界面模式下启动
   WantedBy=graphical.target
   ```

3. **确保X服务器访问权限**：
   允许本地用户访问X服务器：
   ```bash
   xhost +local:
   ```

4. **重新加载并启动服务**：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart test_cv.service
   ```

5. **验证依赖库是否安装**：
   安装可能缺失的库：
   ```bash
   sudo apt install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0
   ```

**关键点解释**：
- `QT_QPA_PLATFORM_PLUGIN_PATH`直接指向Qt的平台插件目录，确保正确加载xcb插件。
- `LD_LIBRARY_PATH`添加conda环境的库路径，避免动态链接库找不到。
- X权限设置确保服务进程有权访问显示服务器。

如果问题依旧，建议：
1. 使用`journalctl -u test_cv.service -f`查看完整日志
2. 在服务中增加`Environment=QT_DEBUG_PLUGINS=1`以获取详细插件加载日志

## 继续解决问题

按照上述流程下来，问题核心是Qt无法连接到X11显示服务器，尽管已配置环境变量。请按以下步骤排查：

- **修复X11权限问题**

```bash
# 确保用户有权限访问X服务器
sudo apt install x11-xserver-utils
xhost +SI:localuser:blues  # 精确授权指定用户
```

- **验证Qt插件路径**

```bash
# 检查真实存在的平台插件路径
ls /home/blues/anaconda3/envs/speech_interaction/lib/python3.8/site-packages/cv2/qt/plugins/platforms
# 应该能看到libqxcb.so

# 修改service文件环境变量
Environment=QT_QPA_PLATFORM_PLUGIN_PATH=/home/blues/anaconda3/envs/speech_interaction/lib/python3.8/site-packages/cv2/qt/plugins
```

- **添加OpenGL库支持**

```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```

- **改进systemd服务配置**

```ini
[Unit]
Description=Test Cv media player.
After=graphical.target systemd-user-sessions.service
Requires=graphical.target

[Service]
Type=simple
ExecStart=/home/blues/anaconda3/envs/speech_interaction/bin/python /path/to/startup.py
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/blues/.Xauthority
Environment=QT_DEBUG_PLUGINS=1
Environment=QT_QPA_PLATFORM=xcb
User=blues
Group=blues
Restart=on-failure

[Install]
WantedBy=graphical.target
```

- **验证X11连接性**

```bash
# 切换到服务用户身份测试
sudo -u blues env DISPLAY=:0 xeyes
```

- **检查conda环境完整性**

```bash
conda activate speech_interaction
ldd /home/blues/anaconda3/envs/speech_interaction/lib/python3.8/site-packages/cv2/qt/plugins/platforms/libqxcb.so | grep "not found"
```

- **关键依赖安装**

```bash
sudo apt install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0
```

- **查看完整日志定位问题**

```bash
journalctl -u test_cv.service -n 100 --no-pager
```

- **最终验证步骤：**

  - 重新加载服务配置

```bash
sudo systemctl daemon-reload
sudo systemctl restart test_cv.service
```
  - 观察实时日志

```bash
journalctl -u test_cv.service -f
```

主要问题在于X11服务器权限和Qt插件路径配置。通过精确指定插件路径、增加X11访问权限、补充图形库依赖，应能解决无法连接显示服务器的问题。如果系统存在多用户登录情况，建议使用虚拟显示方案更稳定。

## 示例代码

- `test_cv.service`

```shell
[Unit]
Description=VLC media player.
After=graphical.target systemd-user-sessions.service
Requires=graphical.target

[Service]
Type=simple
ExecStart=bash /home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/startup.sh
#ExecStart=/home/blues/anaconda3/envs/speech_interaction/bin/python /home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/startup.py
Environment=DISPLAY=:1
#Environment=XAUTHORITY=/home/blues/.Xauthority
# 获取详细插件加载日志
#Environment=QT_DEBUG_PLUGINS=1
#Environment=QT_QPA_PLATFORM=xcb
User=blues
Group=blues
#Restart=on-failure

[Install]
WantedBy=graphical.target
```

- `demo_cv.py`

```python

import time
import os

import cv2


def _emotion_play(filepath):
    # load the corresponding emoji video and play it on the screen
    while True:
        # self.logger.debug(f"*****displaying the emotion video******* {filepath}")
        if os.path.exists(filepath):
            window_name = "window"
            cap = cv2.VideoCapture(filepath)
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            tik = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Display the frame
                cv2.imshow(window_name, frame)
                # Press 'q' to exit the video playback
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            tok = time.time()
            print(f"{filepath.split(os.path.sep)[-1]} emotion video duration: {tok - tik}")
            # Release the video capture object and close all OpenCV windows
            # cap.release()
            # cv2.destroyAllWindows()
        else:
            print(f"The selected emotion file cannot be found: {filepath}")
        time.sleep(0.005)


# 示例使用
if __name__ == "__main__":
    filepath = "/home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/anger10.mp4"
    print(os.environ)
    _emotion_play(filepath)
```

- `demo_vlc.py`

```python
import vlc
import time
import threading


class VideoPlayer:
    def __init__(self):
        self.instance = vlc.Instance('--input-repeat=999999')  # --repeat 不起作用，针对的是播放列表
        self.player = self.instance.media_player_new()
        self.player.set_fullscreen(True)  # 全屏模式
        self.player.video_set_scale(0)    # 自适应画面比例
        self.current_media = None
        self.lock = threading.Lock()

    def play(self, media_path):
        """播放或切换视频源"""
        with self.lock:
            if self.player.is_playing():
                self.player.stop()  # 停止当前播放
            self.current_media = self.instance.media_new(media_path)
            self.player.set_media(self.current_media)
            self.player.play()

    def stop(self):
        """停止播放"""
        with self.lock:
            self.player.stop()


# 示例使用
if __name__ == "__main__":
    filepath = "/home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/anger10.mp4"
    # vlc
    player = VideoPlayer()
    print(1)
    player.play(filepath)  # 初始播放
    print(2)
    # # 模拟 5 秒后动态切换视频源
    # threading.Timer(5, lambda: player.play("shyness100.mp4")).start()
    print(3)
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        player.stop()
```

- `startup.sh`

```shell
#!/bin/bash

/home/blues/anaconda3/envs/speech_interaction/bin/python /home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/startup.py
```

- `startup.py`

```python
import subprocess
import time
import os
env = os.environ.copy()
print("env: ", env)

p = subprocess.Popen(['/home/blues/anaconda3/envs/speech_interaction/bin/python', '/home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/demo_cv.py'],   env=env)
# p = subprocess.Popen(['/home/blues/anaconda3/envs/speech_interaction/bin/python', '/home/blues/vscode_projects/cyan_demos/python_demo/vlc_demo/demo_vlc.py'],   env=env)

# 保持主线程运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
    print("done")
```
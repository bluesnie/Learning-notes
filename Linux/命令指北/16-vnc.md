###### datetime:2025/10/21 17:37:00

###### author:nzb

# VNC

VNC [1] (Virtual Network Console)是虚拟网络控制台的缩写。它 是一款优秀的远程控制工具软件，由著名的 AT&T 的欧洲研究实验室开发的。VNC 是在基于 UNIX 和 Linux 操作系统的免费的开源软件，远程控制能力强大，高效实用，其性能可以和 Windows 和 MAC 中的任何远程控制软件媲美。 在 Linux 中，VNC 包括以下四个命令：vncserver，vncviewer，vncpasswd，和 vncconnect。大多数情况下用户只需要其中的两个命令：vncserver 和 vncviewer。

```shell
1. 安装桌面环境 
# 安装XFCE桌面环境（推荐）  # 选择 lightdm
sudo apt install -y xfce4 xfce4-goodies xorg dbus-x11

2. 配置VNC
# 确保VNC服务器已安装
sudo apt install -y tightvncserver

# 创建VNC配置目录
mkdir -p ~/.vnc
chmod 700 ~/.vnc

# 创建xstartup文件
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
export XKL_XMODMAP_DISABLE=1
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF

chmod +x ~/.vnc/xstartup

3. 设置VNC密码
vncpasswd

4. 启动VNC服务器
# 或者指定更大分辨率
# vncserver :1 -geometry 1920x1080 -depth 24
vncserver :2102 -geometry 1920x1080 -depth 24

端口对应关系：
:1 → 端口 5901
:2 → 端口 5902
:N → 端口 5900 + N

# 关闭特定的VNC会话
vncserver -kill :1
vncserver -kill :2

5. 在本地命令行执行(Windows, Linux, macOS)
# 安装gvncviewer
sudo apt install -y gvncviewer 
gvncviewer 106.54.226.43:2102

6. 通过浏览器访问(安卓设备，使用NoVNC)
  
# 在服务器上安装和启动NoVNC
sudo apt install novnc websockify
vncserver :2102 -geometry 1920x1080 -depth 24  # 确保VNC服务器运行
websockify -D --web=/usr/share/novnc/ 8003 localhost:8002  # 5900 + 2102

# 杀掉 websockify 进程
sudo pkill -f websockify

# 然后在本地浏览器访问
http://106.54.226.43:8003/vnc.html
```

- 设置 `vnc` 开机自启

```shell
sudo vim /etc/systemd/system/vncserver@.service
```

```text
[Unit]
Description=Start TightVNC server at startup
After=syslog.target network.target

[Service]
Type=forking
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu
PIDFile=/home/ubuntu/.vnc/%H:%i.pid
ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver -geometry 1920x1080 -depth 24 :%i
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
```

```shell
sudo systemctl daemon-reload
sudo systemctl enable vncserver@2102.service
sudo systemctl start vncserver@2102.service
```
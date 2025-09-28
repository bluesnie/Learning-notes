###### datetime:2025/09/22 19:35:00

###### author:nzb

# Linux命令之ssh命令

- `ssh -X` 开启X11转发

远程服务器需要安装X11服务

`sudo apt-get install x11-apps`

并设置X11转发

`sudo vi /etc/ssh/sshd_config`
`X11Forwarding yes`

- 测试：`ssh`连接后运行`xclock`命令

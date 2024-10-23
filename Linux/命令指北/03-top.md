###### datetime:2023-02-03 10:50:00

###### author:nzb

## top 命令

### 命令格式

`top [参数]`

### 命令功能

显示当前系统正在执行的进程的相关信息，包括进程ID、内存占用率、CPU占用率等

### 命令参数

| 参数 | 描述 |
| ----- | ----- |
| -b | 批处理 |
| -c | 显示完整的治命令 |
| -I | 忽略失效过程 |
| -s | 保密模式 |
| -S | 累积模式 |
| -i<时间> | 设置间隔时间 |
| -u<用户名> | 指定用户名 |
| -p<进程号> | 指定进程 |
| -n<次数> | 循环显示的次数 |

### 使用实例

#### 1、显示进程信息

- 命令：`top`
- 输出：

```text
[hc@localhost ~]$ top

top - 09:22:56 up 6 days,  1:40,  3 users,  load average: 0.22, 0.31, 0.71
Tasks: 231 total,   1 running, 230 sleeping,   0 stopped,   0 zombie
%Cpu(s): 10.6 us, 12.1 sy,  0.0 ni, 77.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem :  3863568 total,   473100 free,  1651284 used,  1739184 buff/cache
KiB Swap:  3145724 total,  3120012 free,    25712 used.  1837920 avail Mem 

   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND    
  2676 hc        20   0 3627652 339496  22160 S  16.3  8.8  17:50.00 gnome-she+ 
   689 polkitd   20   0  649880  17144   4636 S  10.3  0.4 678:56.95 polkitd    
  2051 root      20   0  358552  50640   6600 S   6.6  1.3   4:44.68 X          
101038 hc        20   0  771640  27384  17392 S   2.7  0.7   0:01.21 gnome-ter+ 
   721 dbus      20   0   61996   3708   1648 S   2.0  0.1 186:05.55 dbus-daem+ 
   680 root      20   0  396404   3816   3124 S   1.7  0.1 154:18.42 accounts-+ 
 86929 root      20   0  457956  51100   6944 S   1.3  1.3  15:59.97 uwsgi      
113983 hc        20   0  161972   2400   1620 R   0.7  0.1   0:00.11 top        
     9 root      20   0       0      0      0 S   0.3  0.0  28:14.62 rcu_sched  
   405 root      20   0       0      0      0 S   0.3  0.0   7:43.36 xfsaild/d+ 
   681 root      20   0   13216    600    572 S   0.3  0.0   0:31.09 rngd       
  1304 mongod    20   0 1025840  81704   4160 S   0.3  2.1  51:29.25 mongod     
  1869 mysql     20   0 1263256 112036   4784 S   0.3  2.9  10:32.82 mysqld     
  2909 hc        20   0  611472   7256   3592 S   0.3  0.2  54:46.85 gsd-accou+ 
 30239 root      20   0  453640  54536   3420 S   0.3  1.4  12:21.71 celery     
     1 root      20   0  125804   3544   2120 S   0.0  0.1   0:16.95 systemd    
     2 root      20   0       0      0      0 S   0.0  0.0   0:00.44 kthreadd   
```

##### 说明

- 统计信息区： 前五行是当前系统情况整体的统计信息区。下面我们看每一行信息的具体意义。

- 第一行，任务队列信息，同 uptime 命令的执行结果，具体参数说明情况如下：
    - 09:22:56 -- 当前系统时间
    - up 6 days, 1:40 -- 系统已经运行了6天1小时40分钟（在这期间系统没有重启过）
    - 3 users -- 当前有2个用户登录系统
    - load average: 0.22, 0.31, 0.71 -- load average后面的三个数分别是1分钟、5分钟、15分钟的负载情况。load
      average数据是每隔5秒钟检查一次活跃的进程数，然后按特定算法计算出的数值。如果这个数除以逻辑CPU的数量，结果高于5的时候就表明系统在超负荷运转了。

- 第二行，Tasks — 任务（进程），具体信息说明如下：系统现在共有231个进程，其中处于运行中的有1个，230个在休眠（sleep），stoped状态的有0个，zombie状态（僵尸）的有0个。

- 第三行，cpu状态信息，具体属性说明如下
    - 10.6 us -- 用户态占用CPU的百分比
    - 12.1 sy -- 内核态占用CPU的百分比
    - 0.0 ni -- 用做nice加权的进程分配的用户态cpu占用CPU的百分比
    - 77.3 id -- 空闲的cpu百分比
    - 0.0 wa -- cpu等待磁盘写入占用CPU的百分比
    - 0.0 hi -- 硬中断（Hardware IRQ）占用CPU的百分比
    - 0.0 si -- 软中断（Software Interrupts）占用CPU的百分比
    - 0.0 st
    - 备注：在这里CPU的使用比率和windows概念不同，需要理解linux系统用户空间和内核空间的相关知识！

- 第四行,内存状态，具体信息如下：
    - 3863568 total -- 物理内存总量
    - 473100 free -- 空闲内存总量
    - 1651284 used -- 使用中的内存总量
    - 1739184 buff/cache -- 缓存的内存量

- 第五行，swap交换分区信息，具体信息说明如下：
    - 3145724 total -- 交换区总量
    - 3120012 free -- 空闲交换区总量
    - 25712 used -- 使用的交换区总量
    - 1837920 avail Mem -- 表示可用于进程下一次分配的物理内存数量

- 第六行，空行

- 第七行以下各进程（任务）的状态监控，项目列信息说明如下：

| 列名 | 说明 |
| ----- | ----- |
| PID | 进程id |
| USER | 进程所有者 |
| PR | 进程优先级 |
| NI | nice值。负值表示高优先级，正值表示低优先级 |
| VIRT | 进程使用的虚拟内存总量，单位kb。VIRT=SWAP+RES |
| RES | 进程使用的、未被换出的物理内存大小，单位kb。RES=CODE+DATA |
| SHR | 共享内存大小，单位kb |
| S | 进程状态。D=不可中断的睡眠状态 R=运行 S=睡眠 T=跟踪/停止 Z=僵尸进程 |
| %CPU | 上次更新到现在的CPU时间占用百分比 |
| %MEM | 进程使用的物理内存百分比 |
| TIME+ | 进程使用的CPU时间总计，单位1/100秒 |
| COMMAND | 进程名称（命令名/命令行） |

- 备注：

    - 第四行中使用中的内存总量（used）指的是现在系统内核控制的内存数，空闲内存总量（free）是内核还未纳入其管控范围的数量。
      纳入内核管理的内存不见得都在使用中，还包括过去使用过的现在可以被重复利用的内存，内核并不把这些可被重新使用的内存交还到free中去， 因此在linux上free内存会越来越少，但不用为此担心。

    - 如果出于习惯去计算可用内存数，这里有个近似的计算公式：`第四行的free + 第四行的buff/cache`，按这个公式此台服务器的可用内存。

    - 对于内存监控，在top里我们要时刻监控第五行swap交换分区的used，如果这个数值在不断的变化，说明内核在不断进行内存和swap的数据交换，这是真正的内存不够用了。

#### 2、显示完整命令

- 命令：`top -c`
- 输出：

```text
[hc@localhost ~]$ top -c

top - 10:01:50 up 6 days,  2:19,  3 users,  load average: 0.01, 0.04, 0.10
Tasks: 233 total,   1 running, 232 sleeping,   0 stopped,   0 zombie
%Cpu(s):  5.3 us, 10.8 sy,  0.0 ni, 83.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem :  3863568 total,   451416 free,  1665668 used,  1746484 buff/cache
KiB Swap:  3145724 total,  3120012 free,    25712 used.  1823504 avail Mem 

   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                              
   689 polkitd   20   0  649880  17144   4636 S  10.0  0.4 682:30.53 /usr/lib/polkit-1/polkitd --no-debug 
  2676 hc        20   0 3646940 359304  22308 S   6.0  9.3  18:02.04 /usr/bin/gnome-shell                 
  2051 root      20   0  352052  44092   6600 S   3.3  1.1   4:50.20 /usr/bin/X :0 -background none -nor+ 
   721 dbus      20   0   61996   3708   1648 S   2.3  0.1 187:04.96 /usr/bin/dbus-daemon --system --add+ 
   680 root      20   0  396404   3816   3124 S   2.0  0.1 155:06.59 /usr/libexec/accounts-daemon         
101038 hc        20   0  772056  27784  17528 S   1.3  0.7   0:03.61 /usr/libexec/gnome-terminal-server   
 86929 root      20   0  457956  51100   6944 S   1.0  1.3  16:30.76 /home/hc/project/envs/autoAweme/bin+ 
  1869 mysql     20   0 1263256 112036   4784 S   0.7  2.9  10:35.24 /usr/libexec/mysqld --basedir=/usr + 
  2909 hc        20   0  611472   7256   3592 S   0.7  0.2  55:04.22 /usr/libexec/gsd-account             
     9 root      20   0       0      0      0 S   0.3  0.0  28:22.21 [rcu_sched]                          
   405 root      20   0       0      0      0 S   0.3  0.0   7:45.11 [xfsaild/dm-0]                       
  2641 hc        20   0   60172   2152   1580 S   0.3  0.1   0:00.27 /bin/dbus-daemon --config-file=/usr+ 
  2889 hc        20   0  797116  12812   6428 S   0.3  0.3   4:06.38 /usr/libexec/gsd-color               
 71994 hc        20   0  162116   2504   1704 R   0.3  0.1   0:00.03 top -c                               
     1 root      20   0  125804   3544   2120 S   0.0  0.1   0:16.97 /usr/lib/systemd/systemd --switched+ 
     2 root      20   0       0      0      0 S   0.0  0.0   0:00.44 [kthreadd]                           
     3 root      20   0       0      0      0 S   0.0  0.0   1:26.89 [ksoftirqd/0]   
```

#### 3、以批处理模式显示程序信息

- 命令：`top -b`

#### 4、以累积模式显示程序信息

- 命令：`top -S`

#### 5、设置信息更新次数

- 命令：`top -n 2`
- 说明：表示更新两次后终止更新显示

#### 6、设置信息更新时间

- 命令：`top -d 3`
- 说明：表示更新周期为3秒

#### 7、显示指定的进程信息

- 命令：`top -p 30568`
- 输出：

```text
[hc@localhost ~]$ top -p 30568

top - 10:04:42 up 6 days,  2:22,  3 users,  load average: 0.26, 0.09, 0.11
Tasks:   1 total,   0 running,   1 sleeping,   0 stopped,   0 zombie
%Cpu(s):  9.7 us, 12.3 sy,  0.0 ni, 77.9 id,  0.0 wa,  0.0 hi,  0.2 si,  0.0 st
KiB Mem :  3863568 total,   451040 free,  1665892 used,  1746636 buff/cache
KiB Swap:  3145724 total,  3120012 free,    25712 used.  1823304 avail Mem 

   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                              
 30568 root      20   0  310244  52392   3652 S   0.0  1.4   0:29.02 uwsgi       
```

### 其他使用技巧

#### 1、多U多核CPU监控

在top基本视图中，按键盘数字“1”，可监控每个逻辑CPU的状况：

- 输出：

```text
top - 09:52:33 up 6 days,  2:10,  3 users,  load average: 0.00, 0.01, 0.11
Tasks: 233 total,   2 running, 230 sleeping,   0 stopped,   1 zombie
%Cpu0  :  2.7 us,  9.2 sy,  0.0 ni, 88.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu1  :  3.4 us, 10.3 sy,  0.0 ni, 86.2 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem :  3863568 total,   458724 free,  1659276 used,  1745568 buff/cache
KiB Swap:  3145724 total,  3120012 free,    25712 used.  1829944 avail Mem 
```

- 说明：观察上图，服务器有2个逻辑CPU，实际上是1个物理CPU。再按数字键1，就会返回到top基本视图界面。

#### 2、高亮显示当前运行进程

在top基本视图中，敲击键盘“b”（打开/关闭加亮效果） 可以通过敲击“y”键关闭或打开运行态进程的加亮效果。

#### 3、进程字段排序

默认进入top时，各进程是按照CPU的占用量来排序的， 敲击键盘“x”（打开/关闭排序列的加亮效果）

#### 4、通过”`shift + >`”或”`shift + <`”可以向右或左改变排序列

#### 5、top交互命令

在top 命令执行过程中可以使用的一些交互命令。这些命令都是单字母的，如果在命令行中使用了`s`选项， 其中一些命令可能会被屏蔽。

| 命令 | 说明 |
| ----- | ----- |
| h | 显示帮助画面，给出一些简短的命令总结说明 |
| k | 终止一个进程。 |
| i | 忽略闲置和僵死进程。这是一个开关式命令。 |
| q | 退出程序 |
| r | 重新安排一个进程的优先级别 |
| S | 切换到累计模式 |
| s | 改变两次刷新之间的延迟时间（单位为s），如果有小数，就换算成m s。输入0值则系统将不断刷新，默认值是5s |
| f或者F | 从当前显示中添加或者删除项目 |
| o或者O | 改变显示项目的顺序 |
| l | 切换显示平均负载和启动时间信息 |
| m | 切换显示内存信息 |
| t | 切换显示进程和CPU状态信息 |
| c | 切换显示命令名称和完整命令行 |
| e | 切换显示内存信息以M、G、T等方式显示 |
| M | 根据驻留内存大小进行排序 |
| P | 根据CPU使用百分比大小进行排序 |
| T | 根据时间/累计时间进行排序 |
| W | 将当前设置写入~/.toprc文件中 |

## btop

[还在用 top htop? 赶紧换 btop 吧，真香！](https://mp.weixin.qq.com/s/Qr-z0-zL44UjnItmDlsMzg)





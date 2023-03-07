###### datetime:2023-02-02 14:20:00

###### author:nzb

## ps 和 pstree 命令

Linux中的`ps`命令是`Process Status`的缩写。`ps`命令用来列出系统中当前运行的那些进程。`ps` 命令列出的是当前那些进程的快照，就是执行`ps`
命令的那个时刻的那些进程，如果想要动态的显示进程信息，就可以使用`top`命令。

要对进程进行监测和控制，首先必须要了解当前进程的情况，也就是需要查看当前进程，而 `ps`
命令就是最基本同时也是非常强大的进程查看命令。使用该命令可以确定有哪些进程正在运行和运行的状态、进程是否结束、进程有没有僵死、哪些进程占用了过多的资源等等。总之大部分信息都是可以通过执行该命令得到的。

`ps` 为我们提供了进程的一次性的查看，它所提供的查看结果并不动态连续的；如果想对进程时间监控，应该用 `top` 工具。

`kill` 命令用于杀死进程。

### 一、命令格式

`ps [参数]`

### 二、命令功能

用于显示当前进程 (process) 的状态。

### 三、命令参数

`ps` 的参数非常多, 在此仅列出几个常用的参数并大略介绍含义

| 参数 | 描述 |
| ----- | ----- |
| -A | 列出所有的行程 |
| -e | 等于“-A” |
| -a | 显示现行终端机下的所有进程，包括其他用户的进程； |
| -u | 以用户为主的进程状态 ； |
| x | 通常与 a 这个参数一起使用，可列出较完整信息。 |
| -w | 显示加宽可以显示较多的资讯 |
| -au | 显示较详细的资讯 |
| -aux | 显示所有包含其他使用者的行程 |
| -f | 做一个更为完整的输出。 |

### 四、使用实例

#### 1、显示所有进程信息

- 命令：`ps -A`

- 输出：

```text
[root@localhost autoAweme]# ps -A
   PID TTY          TIME CMD
     1 ?        00:00:15 systemd
     2 ?        00:00:00 kthreadd
     3 ?        00:00:56 ksoftirqd/0
     5 ?        00:00:00 kworker/0:0H
     7 ?        00:01:01 migration/0
     8 ?        00:00:00 rcu_bh
     9 ?        00:18:57 rcu_sched
    10 ?        00:00:00 lru-add-drain
    11 ?        00:00:03 watchdog/0
    12 ?        00:00:02 watchdog/1
    13 ?        00:01:01 migration/1
    14 ?        00:00:56 ksoftirqd/1
    16 ?        00:00:00 kworker/1:0H
……省略部分结果
```

#### 2、显示指定用户信息

- 命令：`ps -u root`
- 输出：

```text
[root@localhost autoAweme]# ps -u root
   PID TTY          TIME CMD
     1 ?        00:00:15 systemd
     2 ?        00:00:00 kthreadd
     3 ?        00:00:56 ksoftirqd/0
     5 ?        00:00:00 kworker/0:0H
     7 ?        00:01:01 migration/0
     8 ?        00:00:00 rcu_bh
     9 ?        00:18:57 rcu_sched
    10 ?        00:00:00 lru-add-drain
    11 ?        00:00:03 watchdog/0
    12 ?        00:00:02 watchdog/1
    13 ?        00:01:01 migration/1
    14 ?        00:00:56 ksoftirqd/1
    16 ?        00:00:00 kworker/1:0H
    18 ?        00:00:00 kdevtmpfs
    19 ?        00:00:00 netns
    20 ?        00:00:00 khungtaskd
……省略部分结果
```

> 说明：显示root进程用户信息

#### 3、显示所有进程信息，连带命令行

- 命令：`ps -ef`
- 输出：

```text
[root@localhost autoAweme]# ps -ef
UID         PID   PPID  C STIME TTY          TIME CMD
root          1      0  0 11月30 ?      00:00:15 /usr/lib/systemd/systemd --swi
root          2      0  0 11月30 ?      00:00:00 [kthreadd]
root          3      2  0 11月30 ?      00:00:56 [ksoftirqd/0]
root          5      2  0 11月30 ?      00:00:00 [kworker/0:0H]
root          7      2  0 11月30 ?      00:01:01 [migration/0]
……省略部分结果
```

#### 4、ps 与grep 常用组合用法，查找特定进程

- 命令：`ps -ef|grep uwsgi`
- 输出：

```text
[root@localhost autoAweme]# ps -ef|grep uwsgi
root      30568    795  0 12月01 ?      00:00:19 /home/hc/project/envs/pgc/bin/uwsgi --ini /home/hc/project/pgc.ini
root      30578  30568  0 12月01 ?      00:00:00 /home/hc/project/envs/pgc/bin/uwsgi --ini /home/hc/project/pgc.ini
root      66069    795  1 12:07 ?        00:04:29 /home/hc/project/envs/autoAweme/bin/uwsgi --ini /home/hc/project/autoAweme.ini
root      66096  66069  0 12:07 ?        00:00:01 /home/hc/project/envs/autoAweme/bin/uwsgi --ini /home/hc/project/autoAweme.ini
root      80022  86053  0 16:06 pts/1    00:00:00 grep --color=auto uwsgi
```

#### 5、将目前属于您自己这次登入的 PID 与相关信息列示出来

- 命令：`ps -l`
- 输出：

```text
[root@localhost autoAweme]# ps -l
F S   UID    PID   PPID  C PRI  NI ADDR SZ WCHAN  TTY          TIME CMD
4 S     0  85984  80319  0  80   0 - 58596 do_wai pts/1    00:00:00 su
4 S     0  86053  85984  0  80   0 - 29208 do_wai pts/1    00:00:01 bash
0 R     0 107795  86053  0  80   0 - 38300 -      pts/1    00:00:00 ps
```

- 各相关信息的意义

| 标志 | 意义 |
| ----- | ----- |
| F | 代表这个程序的旗标 (flag)， 4 代表使用者为 super user |
| S | 代表这个程序的状态 (STAT)，关于各 STAT 的意义将在内文介绍 |
| UID | 程序被该 UID 所拥有 |
| PID | 就是这个程序的 ID ！ |
| PPID | 则是其上级父程序的ID |
| C | CPU 使用的资源百分比 |
| PRI | 指进程的执行优先权(Priority的简写)，其值越小越早被执行； |
| NI | 这个进程的nice值，其表示进程可被执行的优先级的修正数值。 |
| ADDR | 这个是内核函数，指出该程序在内存的那个部分。如果是个 running的程序，一般就是 "-" |
| SZ | 使用掉的内存大小 |
| WCHAN | 目前这个程序是否正在运作当中，若为 - 表示正在运作 |
| TTY | 登入者的终端机位置 |
| TIME | 使用掉的 CPU 时间。 |
| CMD | 所下达的指令为何 |

在预设的情况下， `ps` 仅会列出与目前所在的 `bash shell` 有关的 `PID` 而已，所以， 当我使用 `ps -l` 的时候，只有三个 `PID`。

#### 6、列出目前所有的正在内存当中的程序

- 命令：`ps aux`
- 输出：

```text
[root@localhost autoAweme]# ps aux
USER        PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root          1  0.0  0.1 125804  4260 ?        Ss   11月30   0:15 /usr/lib/systemd/systemd --switched-root --system --deserialize 22
root          2  0.0  0.0      0     0 ?        S    11月30   0:00 [kthreadd]
root          3  0.0  0.0      0     0 ?        S    11月30   0:56 [ksoftirqd/0]
root          5  0.0  0.0      0     0 ?        S<   11月30   0:00 [kworker/0:0H]
root          7  0.0  0.0      0     0 ?        S    11月30   1:01 [migration/0]
root          8  0.0  0.0      0     0 ?        S    11月30   0:00 [rcu_bh]
root          9  0.3  0.0      0     0 ?        S    11月30  19:02 [rcu_sched]
……省略部分结果
```

- 说明：

| 标志 | 意义 |
| ----- | ----- |
| USER | 该 process 属于那个使用者账号的 |
| PID | 该 process 的号码 |
| %CPU | 该 process 使用掉的 CPU 资源百分比 |
| %MEM | 该 process 所占用的物理内存百分比 |
| VSZ | 该 process 使用掉的虚拟内存量 (Kbytes) |
| RSS | 该 process 占用的固定的内存量 (Kbytes) |
| TTY | 该 process 是在那个终端机上面运作，若与终端机无关，则显示 ?，另外， tty1-tty6 是本机上面的登入者程序，若为 pts/0 等等的，则表示为由网络连接进主机的程序。 |
| STAT | 该程序目前的状态 |
| START | 该 process 被触发启动的时间 |
| TIME | 该 process 实际使用 CPU 运作的时间 |
| COMMAND | 该程序的实际指令 |

- STAT：该程序目前的状态，`ps`工具标识进程的5种状态码
    - D ：不可中断 uninterruptible sleep (usually IO)
    - R ：该程序目前正在运作，或者是可被运作
    - S ：该程序目前正在睡眠当中 (可说是 idle 状态)，但可被某些讯号 (signal) 唤醒。
    - T ：该程序目前正在侦测或者是停止了
    - Z ：该程序应该已经终止，但是其父程序却无法正常的终止他，造成 zombie (疆尸) 程序的状态

#### 7、以类似进程树的结构显示

- 命令：`ps -axjf`
- 输出：

```text
[root@localhost autoAweme]# ps -axjf
  PPID    PID   PGID    SID TTY       TPGID STAT   UID   TIME COMMAND
     0      2      0      0 ?            -1 S        0   0:00 [kthreadd]
     2      3      0      0 ?            -1 S        0   0:57  \_ [ksoftirqd/0]
     2      5      0      0 ?            -1 S<       0   0:00  \_ [kworker/0:0H]
     2      7      0      0 ?            -1 S        0   1:02  \_ [migration/0]
……省略部分结果
     1  80310   2416   2416 ?            -1 Sl    1000   0:25 /usr/libexec/gnome-terminal-server
 80310  80318   2416   2416 ?            -1 S     1000   0:00  \_ gnome-pty-helper
 80310  80319  80319  80319 pts/1     28727 Ss    1000   0:00  \_ bash
 80319  85984  85984  80319 pts/1     28727 S        0   0:00      \_ su
 85984  86053  86053  80319 pts/1     28727 S        0   0:01          \_ bash
 86053  28727  28727  80319 pts/1     28727 R+       0   0:00              \_ ps -axjf
```

#### 8、pstree命令更优雅的树状显示

`pstree`命令以树状图显示进程间的关系（display a tree of processes）。ps命令可以显示当前正在运行的那些进程的信息，但是对于它们之间的关系却显示得不够清晰。在Linux系统中，系统调用`fork`
可以创建子进程， 通过子`shell`也可以创建子进程，Linux系统中进程之间的关系天生就是一棵树，树的根就是进程`PID`为`1`的`init`进程。

- 以树状图只显示进程的名字，且相同进程合并显示
    - 命令：`pstree`
    - 输出：

  ```text
  [root@localhost autoAweme]# pstree
  systemd─┬─ModemManager───2*[{ModemManager}]
          ├─NetworkManager───2*[{NetworkManager}]
          ├─VGAuthService
          ├─2*[abrt-watch-log]
          ├─abrtd
          ├─accounts-daemon───2*[{accounts-daemon}]
          ├─alsactl
          ├─at-spi-bus-laun─┬─dbus-daemon
          │                 └─3*[{at-spi-bus-laun}]
          ├─at-spi2-registr───2*[{at-spi2-registr}]
          ├─atd
          ├─auditd─┬─audispd─┬─sedispatch
          │        │         └─{audispd}
          │        └─{auditd}
          ├─avahi-daemon───avahi-daemon
  ……省略部分结果
  ```

- 以树状图显示进程同时还显示PID
    - 命令：`pstree -p`
    - 输出：

  ```text
  [root@localhost autoAweme]# pstree -p
  systemd(1)─┬─ModemManager(686)─┬─{ModemManager}(722)
             │                   └─{ModemManager}(744)
             ├─NetworkManager(796)─┬─{NetworkManager}(807)
             │                     └─{NetworkManager}(811)
             ├─VGAuthService(677)
             ├─abrt-watch-log(698)
             ├─abrt-watch-log(703)
             ├─abrtd(684)
             ├─accounts-daemon(680)─┬─{accounts-daemon}(699)
             │                      └─{accounts-daemon}(742)
             ├─alsactl(679)
             ├─at-spi-bus-laun(2636)─┬─dbus-daemon(2641)
             │                       ├─{at-spi-bus-laun}(2637)
             │                       ├─{at-spi-bus-laun}(2638)
             │                       └─{at-spi-bus-laun}(2640)
             ├─at-spi2-registr(2643)─┬─{at-spi2-registr}(2648)
             │                       └─{at-spi2-registr}(2649)
             ├─atd(1171)
  ……省略部分结果
  ```

- 以树状图显示进程PID为的进程以及子孙进程，如果有`-p`参数则同时显示每个进程的PID
    - 命令： `pstree [-p] <pid>`
    - 输出：
  ```text
  [root@localhost autoAweme]# pstree 1244
  mysqld_safe───mysqld───19*[{mysqld}]
  [root@localhost autoAweme]# pstree -p 1244
  mysqld_safe(1244)───mysqld(1869)─┬─{mysqld}(1906)
                                   ├─{mysqld}(1911)
                                   ├─{mysqld}(1912)
                                   ├─{mysqld}(1913)
                                   ├─{mysqld}(1914)
                                   ├─{mysqld}(1915)
                                   ├─{mysqld}(1916)
                                   ├─{mysqld}(1917)
                                   ├─{mysqld}(1918)
                                   ├─{mysqld}(1919)
                                   ├─{mysqld}(1920)
                                   ├─{mysqld}(1926)
                                   ├─{mysqld}(1927)
                                   ├─{mysqld}(1928)
                                   ├─{mysqld}(1929)
                                   ├─{mysqld}(1930)
                                   ├─{mysqld}(1931)
                                   ├─{mysqld}(2081)
                                   └─{mysqld}(77714)
  ```

- 以树状图显示进程，相同名称的进程不合并显示，并且会显示命令行参数，如果有`-p`参数则同时显示每个进程的PID
    - 命令：`pstree -a`
    - 输出：

  ```text
  [root@localhost autoAweme]# pstree -a
  systemd --switched-root --system --deserialize 22
    ├─ModemManager
    │   └─2*[{ModemManager}]
    ├─NetworkManager --no-daemon
    │   └─2*[{NetworkManager}]
    ├─VGAuthService -s
    ├─supervisord /usr/bin/supervisord -c /etc/supervisord.conf
    │   ├─celery /home/hc/project//envs/autoAweme/bin/celery worker -A celery_worker.celery -l info
    │   │   ├─celery /home/hc/project//envs/autoAweme/bin/celery worker -A celery_worker.celery -l info
    │   │   │   └─{celery}
    │   │   ├─celery /home/hc/project//envs/autoAweme/bin/celery worker -A celery_worker.celery -l info
    │   │   │   └─{celery}
    │   │   └─2*[{celery}]
    │   ├─uwsgi --ini /home/hc/project/pgc.ini
    │   │   └─uwsgi --ini /home/hc/project/pgc.ini
    │   └─uwsgi --ini /home/hc/project/autoAweme.ini
    │       ├─uwsgi --ini /home/hc/project/autoAweme.ini
    │       └─2*[{uwsgi}]
  ……省略部分结果
  ```

> **注**：因为pstree输出的信息可能比较多，所以最好与more/less配合使用,使用上下箭头查看，按q退出。
>
>`pstree -p | less`

#### 9、其他实例

- 可以用 | 管道和 more 连接起来分页查看
    - 命令：`ps -aux |more`

- 把所有进程显示出来，并输出到ps001.txt文件
    - 命令：`ps -aux > ps001.txt`

- 输出指定的字段
    - 命令：`ps -o pid,ppid,pgrp,session,tpgid,comm`

### Linux上进程的几种状态

#### R（TASK_RUNNING），可执行状态&运行状态（在run_queue队列里的状态）

只有在该状态的进程才可能在CPU上运行，同一时刻可能有多个进程处于可执行状态，这些进程的`task_struct`结构（进程控制块）被放入对应的CPU的可执行队列中
（一个进程最多只能出现在一个CPU的可执行队列中）。进程调度器的任务就是从各个CPU的可执行队列中分别选择一个进程在该CPU上运行。

一般将正在CPU上执行的进程定义为`RUNNING`状态，而将可执行但是尚未被调度执行的进程定义为`READY`状态，这两种状态在linux下同一为`TASK_RUNNING`状态。
只要可执行队列不为空，其对应的CPU就不能偷懒，就要执行其中某个进程。一般称此时的CPU“忙碌”。对应的，CPU“空闲”就是指其对应的可执行队列为空， 以致于CPU无事可做。

> 有人问，为什么死循环程序会导致CPU占用高呢？因为死循环程序基本上总是处于TASK_RUNNING状态（进程处于可执行队列中）。

#### S（TASK_INTERRUPTIBLE）,可中断的睡眠状态，可处理signal

处于这个状态的进程因为等待某个事件的发生（比如等待socket连接、等待信号量），而被挂起。这些进程的`task_struct`结构被放入对应事件的等待队列中。
当这些事件发生时（由外部中断触发、或由其他进程触发），对应的等待队列中的一个或多个进程被唤醒。通过`ps`命令我们会看到，一般情况下， 进程列表中的绝大多数进程都处于`TASK_INTERRUPTIBLE`
状态（除非机器的负载很高）。毕竟CPU就那么几个，而进程动辄几十上百个， 如果不是绝大多数进程都在睡眠，CPU又怎么响应的过来。

#### D（TASK_UNINTERRUPTIBLE），不可中断的睡眠状态，可处理signal，有延迟

与`TASK_INTERRUPTIBLE`状态类似，进程也处于睡眠状态，但是此刻的进程是不可中断的。**不可中断，指的并不是CPU不响应外部硬件的中断， 而是指进程不响应异步信号**
。绝大多数情况下，进程处在睡眠状态时，总是应该能够响应异步信号的。否则你将惊奇的发现，kill -9竟然杀不死一个正在睡眠的进程了！ 于是我们也很好理解，为什么`ps`命令看到的进程几乎不会出现`TASK_UNINTERRUPTIBLE`
状态，而总是`TASK_INTERRUPTIBLE`状态。

**而`TASK_UNINTERRUPTIBLE状`态存在的意义就在于，内核的某些处理流程是不能被打断的**。如果响应异步信号，
程序的执行流程中就会被插入一段用于处理异步信号的流程（这个插入流程可能只存在于内核态，也可能延伸到用户态），于是原有的流程被中断了。 （参见《linux内核异步中断浅析》）在进程对某些硬件进行操作时（比如进程调用`read`
系统调用对某个设备文件进行读操作， 而`read`系统调用最终执行到对应设备驱动的代码，并与对应的物理设备进行交互），可能需要使用`TASK_UNINTERRUPTIBLE`状态对进程进行保护，
以避免进程与设备交互的过程被打断，造成设备陷入不可控的状态。这种情况下的`TASK_UNINTERRUPTIBLE`状态总是非常短暂的，通过`ps`命令基本上不可能捕捉到。

#### T（TASK_STOPPED or TASK_TRACED）,暂停状态或跟踪状态，不可处理signal,因为根本没有时间片运行代码

向进程发送一个`SIGSTOP`信号，它就会因响应信号而进入`TASK_STOPPED`状态（除非该进程本身处于`TASK_UNINTERRUPTIBLE`状态而不响应信号）。
(`SIGSTOP`与`SIGKILL`信号一样，是非强制的。不允许用户进程通过`signal`系统的系统调用重新设置对应的信号处理函数)向进程发送一个`SIGCONT`信号， 可以让其从`TASK_STOPPED`
状态恢复到`TASK_RUNNING`状态。

当进程正在被跟踪时，它处于`TASK_TRACED`这个特殊的状态。“正在被跟踪”指的是进程暂停下来，等待跟踪它的进程对它进行操作。 比如在`gdb`中对被跟踪的进程下一个断点，**进程在断点处停下来的时候就处于`TASK_TRACED`
状态**。而在其他时候，被跟踪的进程还是处于前面提到的那些状态。 对于进程本身来说，`TASK_STOPPED`和`TASK_TRACED`状态很类似，都是表示进程暂停下来。而`TASK_TRACED`
状态相当于在`TASK_STOPPED`之上多了一层保护，
**处于`TASK_TRACED`状态的进程不能响应`SIGCONT`信号而被唤醒**。只能等到调试进程通过`ptrace`系统调用执行`PTRACE_CONT`、`PTRACE_DETACH`等操作 （通过`ptrace`
系统调用的参数指定操作），或调试进程退出，被调试的进程才能恢复`TASK_RUNNING`状态。

#### Z（TASK_DEAD-EXIT_ZOMBIE）退出状态，进程称为僵尸进程，不可被kill，即不相应任务信号，无法用SIGKILL杀死

在退出过程中，进程占有的所有资源将被回收，除了`task_struct`结构（以及少数资源）以外。于是进程就只剩下`task_struct`这么个空壳，故称为僵尸。
**之所以保留`task_struct`，是因为`task_struct`里面保存了进程的退出码、以及一些统计信息。而其父进程很可能会关心这些信息**。比如在shell中，
$?变量就保存了最后一个退出的前台进程的退出码，而这个退出码往往被作为if语句的判断条件。当然，内核也可以将这些信息保存在别的地方， 而将`task_struct`释放掉，以节省一些空间。但是使用`task_struct`
结构更为方便，因为内核中已经建立了从`pid`到`task_struct`查找关系， 还有进程间的父子关系。释放掉`task_struct`，则需要建立一些新的数据结构，以便让父进程找到它的子进程的退出信息。
**父进程可以通过`wait`系列的系统调用（如`wait4`,`waitid`）来等待某个或某些子进程的退出，并获取它的退出信息。 然后`wait`系列的系统调用会顺便将子进程的尸体(`task_struct`)也释放掉**
。子进程在退出的过程中，内核会给其父进程发送一个信号，通知父进程来收尸。 这个信号默认是`SIGCHLD`，但是在通过`clone`系统调用创建子进程时，可以设置这个信号。只要父进程不退出，这个僵尸状态的子进程就一直存在。
那么如果父进程退出了呢，谁又来给子进程“收尸”？当进程退出的时候，会将它的所有子进程都托管给别的进程（使之成为别的进程的子进程）。
**托管给谁呢？可能是退出进程所在进程组的下一个进程（如果存在的话），或者是1号进程**。所以每个进程、每时每刻都有父进程存在。除非它是1号进程。 1号进程，pid为1的进程，又称`init`进程。

linux系统启动后，第一个被创建的用户态进程就是`init`进程。它有两项使命：

- 1、执行系统初始化脚本，创建一系列的进程（它们都是`init`进程的子孙）；
- 2、在一个死循环中等待其子进程的退出事件，并调用`waitid`系统调用来完成“收尸”工作； `init`进程不会被暂停、也不会被杀死（这是由内核来保证的）。 它在等待子进程退出的过程中处于`TASK_INTERRUPTIBLE`状态，
  “收尸”过程中则处于`TASK_RUNNING`状态。

#### X（TASK_DEAD-EXIT_DEAD），退出状态，进程即将被销毁

而进程在退出过程中也可能不会保留它的`task_struct`。**比如这个进程是多线程程序中被`detach`过的进程**（进程？线程？参见《linux线程浅析》）。 或者**父进程通过设置`SIGCHLD`信号的`handler`
为`SIG_IGN`，显式的忽略了`SIGCHLD`信号**。（这是posix的规定， 尽管子进程的退出信号可以被设置为`SIGCHLD`以外的其他信号。）此时，进程将被置于`EXIT_DEAD`
退出状态，这意味着接下来的代码立即就会将该进程彻底释放。 所以`EXIT_DEAD`状态是非常短暂的，几乎不可能通过`ps`命令捕捉到。

#### 进程的初始状态

进程是通过`fork`系列的系统调用（`fork`、`clone`、`vfork`）来创建的，内核（或内核模块）也可以通过`kernel_thread`函数创建内核进程。
这些创建子进程的函数本质上都完成了相同的功能——将调用进程复制一份，得到子进程。（可以通过选项参数来决定各种资源是共享、还是私有。） 那么既然调用进程处于`TASK_RUNNING`
状态（否则，它若不是正在运行，又怎么进行调用？），则子进程默认也处于`TASK_RUNNING`状态。 另外，在系统调用调用`clone`和内核函数`kernel_thread`也接受`CLONE_STOPPED`
选项，从而将子进程的初始状态置为`TASK_STOPPED`。

#### 进程状态变迁

进程自创建以后，状态可能发生一系列的变化，直到进程退出。而尽管进程状态有好几种，但是进程状态的变迁却只有两个方向——**从`TASK_RUNNING`状态变为`非TASK_RUNNING`状态、 或者从`非TASK_RUNNING`
状态变为`TASK_RUNNING`状态**。也就是说，如果给一个`TASK_INTERRUPTIBLE`状态的进程发送`SIGKILL`信号，这个进程将先被唤醒（进入`TASK_RUNNING`状态）， 然后再响应`SIGKILL`
信号而退出（变为`TASK_DEAD`状态）。**并不会从`TASK_INTERRUPTIBLE`状态直接退出（至少发送一个`SIGCHLD`信号需要活着吧）**。

进程从`非TASK_RUNNING`状态变为`TASK_RUNNING`状态，是由别的进程（也可能是中断处理程序）执行唤醒操作来实现的。 执行唤醒的进程设置被唤醒进程的状态为`TASK_RUNNING`，然后将其`task_struct`
结构加入到某个CPU的可执行队列中。于是被唤醒的进程将有机会被调度执行。

而进程从`TASK_RUNNING`状态变为`非TASK_RUNNING`状态，则有两种途径：

- 1、响应信号而进入`TASK_STOPED`状态、或`TASK_DEAD`状态；
- 2、执行系统调用主动进入`TASK_INTERRUPTIBLE`状态（如`nanosleep`系统调用）、或`TASK_DEAD`状态（如`exit`系统调用）；或由于执行系统调用需要的资源得不到满足，
  而进入`TASK_INTERRUPTIBLE`状态或`TASK_UNINTERRUPTIBLE`状态（如`select`系统调用）。显然，这两种情况都只能发生在进程正在CPU上执行的情况下。
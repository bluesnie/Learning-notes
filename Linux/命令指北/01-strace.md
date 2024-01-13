###### datetime:2022-12-09 15:42:00

###### author:nzb

## strace 命令详解

### 一、strace 是什么？

按照 strace 官网的描述，strace 是一个可用于诊断、调试和教学的 Linux 用户空间跟踪器。我们用它来监控用户空间进程和内核的交互，比如系统调用、信号传递、进程状态变更等。

strace 底层使用内核的 ptrace 特性来实现其功能。

在运维的日常工作中，故障处理和问题诊断是个主要的内容，也是必备的技能。strace 作为一种动态跟踪工具，能够帮助运维高效地定位进程和服务故障。它像是一个侦探，通过系统调用的蛛丝马迹，告诉你异常的真相。

### 二、strace 能做什么？

运维工程师都是实践派的人，我们还是先来个例子吧。

我们从别的机器 copy 了个叫做 some_server 的软件包过来，开发说直接启动就行，啥都不用改。可是尝试启动时却报错，根本起不来！

启动命令：

```text
./some_server ../conf/some_server.conf
```

输出:

```text
FATAL: InitLogFile failed iRet: -1!
Init error: -1655
```

为什么起不来呢？从日志看，似乎是初始化日志文件失败，真相到底怎样呢？我们用 strace 来看看。

```text
strace -tt -f ./some_server ../conf/some_server.conf
```

我们注意到，在输出 InitLogFile failed 错误的前一行，有个 open 系统调用:

```text
23:14:24.448034 open("/usr/local/apps/some_server/log//server_agent.log", O_RDWR|O_CREAT|O_APPEND|O_LARGEFILE, 0666) = -1 ENOENT (No such file or directory)
```

它尝试打开文件 /usr/local/apps/some_server/log//server_agent.log 来写(不存在则创建)，可是却出错了，返回码是 -1 , 系统错误号 errorno 为 ENOENT。 查下 open
系统调用的手册页：`man 2 open` 搜索 ENOENT 这个错误号 errno 的解释

```text
ENOENT O_CREAT  is not set and the named file does not exist.  Or, a directory component in pathname does not exist or is a dangling symbolic link.
```

这里说得比较清楚，因为我们例子中的 open 选项指定了 O_CREAT 选项，这里 errno 为 ENOENT 的原因是日志路径中某个部分不存在或者是一个失效的符号链接。我们来一级一级看下路径中的哪部分不存在：

```text
ls -l /usr/local/apps/some_server/log
ls: cannot access /usr/local/apps/some_server/log: No such file or directory
ls -l /usr/local/apps/some_server
total 8
drwxr-xr-x 2 root users 4096 May 14 23:13 bin
drwxr-xr-x 2 root users 4096 May 14 22:48 conf
```

原来是 log 子目录不存在！上层目录都是存在的。手工创建 log 子目录后，服务就能正常启动了。

回过头来， strace 究竟能做什么呢？

> 它能够打开应用进程的这个黑盒，通过系统调用的线索，告诉你进程大概在干嘛。

### 三、strace怎么用？

strace 有两种运行模式。

一种是通过它启动要跟踪的进程。用法很简单，在原本的命令前加上 strace 即可。比如我们要跟踪 "ls -lh /var/log/messages" 这个命令的执行，可以这样：

```text
strace ls -lh /var/log/messages
```

另外一种运行模式，是跟踪已经在运行的进程，在不中断进程执行的情况下，理解它在干嘛。 这种情况，给 strace 传递个 -p pid 选项即可。比如，有个在运行的 some_server 服务，第一步，查看 pid:

```text

pidof some_server                      
17553

# 查看进程中线程的CPU占用情况，在top中加入`-H`参数，查看该进程中线程的cpu占用情况
top -H -p 17553
```

得到其 pid 17553 然后就可以用 strace 跟踪其执行:

```text
strace -p 17553
```

完成跟踪时，按 Ctrl + C 结束 strace 即可。

strace 有一些选项可以调整其行为，我们这里介绍下其中几个比较常用的，然后通过示例讲解其实际应用效果。

strace 常用选项：

从一个示例命令来看：

```text
strace -tt -T -v -f -e trace=file -o /data/log/strace.log -s 1024 -p 23489
```

- `-tt`：在每行输出的前面，显示毫秒级别的时间
- `-T`：显示每次系统调用所花费的时间
- `-v`：对于某些相关调用，把完整的环境变量，文件 stat 结构等打出来。
- `-f`：跟踪目标进程，以及目标进程创建的所有子进程
- `-e`：控制要跟踪的事件和跟踪行为，比如指定要跟踪的系统调用名称
    - `-e trace=file` 跟踪和文件访问相关的调用(参数中有文件名)
    - `-e trace=process` 和进程管理相关的调用，比如fork/exec/exit_group
    - `-e trace=network` 和网络通信相关的调用，比如socket/sendto/connect
    - `-e trace=signal` 信号发送和处理相关，比如kill/sigaction
    - `-e trace=desc` 和文件描述符相关，比如write/read/select/epoll等
    - `-e trace=ipc` 跟踪所有与进程通讯有关的系统调用，比如shmget等
    - `-e signal=`  指定跟踪的系统信号.默认为all.如 signal=!SIGIO(或者signal=!io),表示不跟踪SIGIO信号.
    - `-e trace=` 只跟踪指定的系统 调用.例如:-e trace=open,close,read,write表示只跟踪这四个系统调用.默认的为set=all.
- `-o`：把 strace 的输出单独写到指定的文件
- `-s`：当系统调用的某个参数是字符串时，最多输出指定长度的内容，默认是 32 个字节
- `-p`：指定要跟踪的进程 pid，要同时跟踪多个 pid，重复多次 -p 选项即可。
- `-x`：打印十六进制非ascii字符串

```text
# 下位机串口数据
# 命令：strace -s 1024 -p 214 -e trace=read -x

read(64, "R", 1)                        = 1
read(64, "U", 1)                        = 1     // 帧尾
read(64, "\xaa", 1)                     = 1     // 帧头
read(64, "L", 1)                        = 1
read(64, "M", 1)                        = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "u", 1)                        = 1
read(64, "6", 1)                        = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x84", 1)                     = 1
read(64, "\xc1", 1)                     = 1
read(64, "#", 1)                        = 1
read(64, "\x86", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x0e", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "d", 1)                        = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x04", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "R", 1)                        = 1
read(64, "U", 1)                        = 1     // 帧尾
read(64, "\xaa", 1)                     = 1     // 帧头
read(64, "L", 1)                        = 1
read(64, "M", 1)                        = 1
```

- `-xx`：打印十六进制ascii字符串（可读性高点）

```text
# 下位机串口数据
# 命令：strace -s 1024 -p 214 -e trace=read -xx

read(64, "\x52", 1)                     = 1
read(64, "\x55", 1)                     = 1     // 帧尾
read(64, "\xaa", 1)                     = 1     // 帧头
read(64, "\x4c", 1)                     = 1
read(64, "\x4d", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x75", 1)                     = 1
read(64, "\x36", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x84", 1)                     = 1
read(64, "\xc1", 1)                     = 1
read(64, "\x23", 1)                     = 1
read(64, "\x86", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x01", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x0e", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x64", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x04", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x00", 1)                     = 1
read(64, "\x53", 1)                     = 1
read(64, "\x55", 1)                     = 1     // 帧尾
read(64, "\xaa", 1)                     = 1     // 帧头
read(64, "\x4c", 1)                     = 1
read(64, "\x4d", 1)                     = 1
```

- `-c`：将进程所有系统调用做一个统计分析并返回

```text
# strace -f -c -p 126

% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 55.87    9.992260         363     27527      1649 futex
 19.84    3.547689        8717       407           pselect6
 11.15    1.994709        1471      1356           ppoll
  6.01    1.074210        5838       184           epoll_pwait
  3.89    0.695708          88      7934       625 recvfrom
  1.45    0.260006          54      4772           fcntl
  0.66    0.117486         474       248           read
  0.65    0.117000       58500         2           accept
  0.19    0.033846          54       626           sendto
  0.07    0.013000         813        16           fsync
  0.06    0.011512          27       421           newfstatat
  0.05    0.008677           7      1327           getpid
  0.04    0.006500          22       298           munmap
  0.03    0.004798          51        94           write
  0.02    0.004247          18       241           mmap
  0.01    0.002207         130        17           openat
  0.01    0.002000          56        36           fstat
  0.00    0.000000           0        19           close
  0.00    0.000000           0        19           lseek
  0.00    0.000000           0         1           mprotect
------ ----------- ----------- --------- --------- ----------------
100.00   17.885855                 45545      2274 total
```

### 四、strace问题定位案例

#### 1、定位进程异常退出

问题：机器上有个叫做run.sh的常驻脚本，运行一分钟后会死掉。需要查出死因。

定位：进程还在运行时，通过ps命令获取其pid, 假设我们得到的pid是24298

```text
strace -o strace.log -tt -p 24298
```

查看 strace.log，我们在最后 2 行看到如下内容:

```text
22:47:42.803937 wait4(-1,  <unfinished ...>
22:47:43.228422 +++ killed by SIGKILL +++
```

这里可以看出，进程是被其他进程用 KILL 信号杀死的。

实际上，通过分析，我们发现机器上别的服务有个监控脚本，它监控一个也叫做 run.sh 的进程，当发现 run.sh 进程数大于 2 时，就会把它杀死重启。结果导致我们这个 run.sh 脚本被误杀。

进程被杀退出时，strace 会输出 killed by SIGX（SIGX 代表发送给进程的信号）等，那么，进程自己退出时会输出什么呢？

这里有个叫做 test_exit 的程序，其代码如下:

```text
#include <stdio.h>
#include <stdlib.h>
 
int main(int argc, char **argv) {
       exit(1);
}
```

我们 strace 看下它退出时 strace 上能看到什么痕迹。

```text
strace -tt -e trace=process -f ./test_exit
```

> 说明: -e trace=process 表示只跟踪和进程管理相关的系统调用。

输出：

```text
23:07:24.672849 execve("./test_exit", ["./test_exit"], [/* 35 vars */]) = 0
23:07:24.674665 arch_prctl(ARCH_SET_FS, 0x7f1c0eca7740) = 0
23:07:24.675108 exit_group(1)           = ?
23:07:24.675259 +++ exited with 1 +++
```

可以看出，进程自己退出时（调用 exit 函数，或者从 main 函数返回）, 最终调用的是 exit_group 系统调用， 并且 strace 会输出 exited with X（X为退出码）。

可能有人会疑惑，代码里面明明调用的是 exit，怎么显示为 exit_group?

> 这是因为这里的 exit 函数不是系统调用，而是 glibc 库提供的一个函数，exit 函数的调用最终会转化为 exit_group 系统调用，它会退出当前进程的所有线程。实际上，有一个叫做 _exit()的系统调用（注意 exit 前面的下划线)，线程退出时最终会调用它。

#### 2、定位共享内存异常

有个服务启动时报错：

```text
shmget 267264 30097568: Invalid argument
Can not get shm...exit!
```

错误日志大概告诉我们是获取共享内存出错，通过 strace 看下：

```text
strace -tt -f -e trace=ipc ./a_mon_svr     ../conf/a_mon_svr.conf
```

输出：

```text
22:46:36.351798 shmget(0x5feb, 12000, 0666) = 0
22:46:36.351939 shmat(0, 0, 0)          = ?
Process 21406 attached
22:46:36.355439 shmget(0x41400, 30097568, 0666) = -1 EINVAL (Invalid argument)
shmget 267264 30097568: Invalid argument
Can not get shm...exit!
```

这里，我们通过 -e trace=ipc 选项，让 strace 只跟踪和进程通信相关的系统调用。

从 strace 输出，我们知道是 shmget 系统调用出错了，errno 是 EINVAL。同样， 查询下 shmget 手册页，搜索 EINVAL 的错误码的说明:

> EINVAL A new segment was to be created and size < SHMMIN or size > SHMMAX, or no new segment was to be created, a segment with given key existed, but size is greater than the size of that segment

翻译下，shmget 设置 EINVAL 错误码的原因为下列之一：

- 要创建的共享内存段比 SHMMIN 小 (一般是1个字节)

- 要创建的共享内存段比 SHMMAX 大 (内核参数 kernel.shmmax 配置)

- 指定 key 的共享内存段已存在，其大小和调用 shmget 时传递的值不同。

从 strace 输出看，我们要连的共享内存 key 0x41400，指定的大小是 30097568 字节，明显与第1、2 种情况不匹配。那只剩下第三种情况。使用 ipcs 看下是否真的是大小不匹配：

```text
ipcs  -m | grep 41400
key        shmid      owner      perms      bytes      nattch     status    
0x00041400 1015822    root       666        30095516   1
```

可以看到，已经 0x41400 这个 key 已经存在，并且其大小为 30095516 字节，和我们调用参数中的 30097568 不匹配，于是产生了这个错误。

在我们这个案例里面，导致共享内存大小不一致的原因，是一组程序中，其中一个编译为32位，另外一个编译为64位,代码里面使用了long这个变长int数据类型。

> 把两个程序都编译为64解决了这个问题。

这里特别说下 strace 的 -e trace 选项。

要跟踪某个具体的系统调用，-e trace=xxx 即可。但有时候我们要跟踪一类系统调用，比如所有和文件名有关的调用、所有和内存分配有关的调用。

如果人工输入每一个具体的系统调用名称，可能容易遗漏。于是strace提供了几类常用的系统调用组合名字。

> - -e trace=file 跟踪和文件访问相关的调用(参数中有文件名)
> - -e trace=process 和进程管理相关的调用，比如fork/exec/exit_group
> - -e trace=network 和网络通信相关的调用，比如socket/sendto/connect
> - -e trace=signal 信号发送和处理相关，比如kill/sigaction
> - -e trace=desc 和文件描述符相关，比如write/read/select/epoll等
> - -e trace=ipc 进程见同学相关，比如shmget等

绝大多数情况，我们使用上面的组合名字就够了。实在需要跟踪具体的系统调用时，可能需要注意C 库实现的差异。

> 比如我们知道创建进程使用的是 fork 系统调用，但在 glibc 里面，fork 的调用实际上映射到了更底层的 clone 系统调用。使用 strace 时，得指定 -e trace=clone，指定 -e trace=fork 什么也匹配不上。

#### 3、 性能分析

假如有个需求，统计 Linux 4.5.4 版本内核中的代码行数（包含汇编和 C 代码）。这里提供两个Shell 脚本实现：

poor_script.sh:

```shell
#!/bin/bash
total_line=0
while read filename; do
   line=$(wc -l $filename | awk '{print $1}')
   (( total_line += line ))
done < <( find linux-4.5.4 -type f  ( -iname '.c' -o -iname '.h' -o -iname '*.S' ) )
echo "total line: $total_line"
 
```

good_script.sh:

```shell
#!/bin/bash
find linux-4.5.4 -type f  ( -iname '.c' -o -iname '.h' -o -iname '*.S' ) -print0 | wc -l —files0-from - | tail -n 1
```

两段代码实现的目的是一样的。 我们通过 strace 的 -c 选项来分别统计两种版本的系统调用情况和其所花的时间（使用 -f 同时统计子进程的情况）

从两个输出可以看出，good_script.sh 只需要 2 秒就可以得到结果：19613114 行。它大部分的调用（calls）开销是文件操作（read/open/write/close）等，统计代码行数本来就是干这些事情。

而 poor_script.sh 完成同样的任务则花了 539 秒。它大部分的调用开销都在进程和内存管理上(wait4/mmap/getpid…)。

实际上，从两个图中 clone 系统调用的次数，我们可以看出 good_script.sh 只需要启动 3 个进程，而 poor_script.sh 完成整个任务居然启动了 126335 个进程！

而进程创建和销毁的代价是相当高的，性能不差才怪。

#### 五、总结

当发现进程或服务异常时，我们可以通过 strace 来跟踪其系统调用，“看看它在干啥”，进而找到异常的原因。熟悉常用系统调用，能够更好地理解和使用strace。

当然，万能的 strace 也不是真正的万能。当目标进程卡死在用户态时，strace 就没有输出了。

这个时候我们需要其他的跟踪手段，比如 gdb / perf / SystemTap 等。

备注：

- 1、perf 原因 kernel 支持

- 2、ftrace kernel 支持可编程

- 3、systemtap 功能强大，RedHat 系统支持，对用户态，内核态逻辑都能探查，使用范围更广。


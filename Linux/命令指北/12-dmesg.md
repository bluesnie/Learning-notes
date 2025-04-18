###### datetime:2025/04/18 11:15:00

###### author:nzb

# Linux命令之dmesg命令

## 一、命令简介
  `Linux dmesg`（英文全称：`display message`）命令用于显示开机信息。 `kernel` 会将开机信息存储在 `ring buffer` 中。您若是开机时来不及查看信息，可利用 `dmesg` 来查看。开机信息亦保存在 `/var/log` 目录中，名称为 `dmesg` 的文件里。

## 二、使用示例

- 1、查看命令版本

```bash
(base) [root@s186 ~]# dmesg -V
dmesg，来自 util-linux 2.23.2
```

- 2、获取命令帮助

```bash
(base) [root@s186 ~]# dmesg -help
用法：
dmesg [选项]

选项：
-C, --clear 清除内核环形缓冲区(ring butter)
-c, --read-clear 读取并清除所有消息
-D, --console-off 禁止向终端打印消息
-d, --show-delta 显示打印消息之间的时间差
-e, --reltime 以易读格式显示本地时间和时间差
…
```

- 3、查看所有开机日志信息

```bash
(base) [root@s186 ~]# dmesg
[ 0.000000] microcode: microcode updated early to revision 0x25, date = 2018-04-02
[ 0.000000] Initializing cgroup subsys cpuset
[ 0.000000] Initializing cgroup subsys cpu
[ 0.000000] Initializing cgroup subsys cpuacct
[ 0.000000] Linux version 3.10.0-957.5.1.el7.x86_64 (mockbuild@kbuilder.bsys.centos.org) (gcc version 4.8.5 20150623 (Red Hat 4.8.5-36) (GCC) ) #1
SMP Fri Feb 1 14:54:57 UTC 2019
[ 0.000000] Command line: BOOT_IMAGE=/vmlinuz-3.10.0-957.5.1.el7.x86_64 root=UUID=062b2cf4-b789-4eb4-bc1a-4b48d8266d16 ro crashkernel=auto biosdev
name=0 rhgb quiet LANG=en_US.UTF-8
[ 0.000000] e820: BIOS-provided physical RAM map:
[ 0.000000] BIOS-e820: [mem 0x0000000000000000-0x000000000009d7ff] usable
…
```

- 4、过滤想查看信息

```bash
#建议使用-i参数过滤时忽略大小写
(base) [root@s186 ~]# dmesg |grep -i cpu
[ 0.000000] Initializing cgroup subsys cpuset
[ 0.000000] Initializing cgroup subsys cpu
[ 0.000000] Initializing cgroup subsys cpuacct
[ 0.000000] ACPI: SSDT 00000000d8ffa810 00539 (v01 PmRef Cpu0Ist 00003000 INTL 20120711)
[ 0.000000] ACPI: SSDT 00000000d8ffad50 00AD8 (v01 PmRef CpuPm 00003000 INTL 20120711)
[ 0.000000] smpboot: Allowing 4 CPUs, 0 hotplug CPUs
[ 0.000000] setup_percpu: NR_CPUS:5120 nr_cpumask_bits:4 nr_cpu_ids:4 nr_node_ids:1
[ 0.000000] PERCPU: Embedded 38 pages/cpu @ffff99c25fa00000 s118784 r8192 d28672 u524288
[ 0.000000] pcpu-alloc: s118784 r8192 d28672 u524288 alloc=1*2097152
[ 0.000000] pcpu-alloc: [0] 0 1 2 3
[ 0.000000] SLUB: HWalign=64, Order=0-3, MinObjects=0, CPUs=4, Nodes=1
[ 0.000000] RCU restricting CPUs from NR_CPUS=5120 to nr_cpu_ids=4.
[ 0.002975] mce: CPU supports 7 MCE banks
[ 0.002981] CPU0: Thermal monitoring enabled (TM1)
[ 0.039076] smpboot: CPU0: Intel® Core™ i3-4130 CPU @ 3.40GHz (fam: 06, model: 3c, stepping: 03)
[ 0.042323] NMI watchdog: enabled on all CPUs, permanently consumes one hw-PMU counter.
[ 0.044633] Brought up 4 CPUs
[ 0.070551] ACPI: SSDT ffff99c25fc62400 003D3 (v01 PmRef Cpu0Cst 00003001 INTL 20120711)
[ 0.341801] ACPI: Requesting acpi_cpufreq
[ 0.369369] cpuidle: using governor menu
[ 7.501018] cryptd: max_cpu_qlen set to 1000
```

- 5、便于阅读的方式显示日志日期和时间

```bash
(base) [root@s186 ~]# dmesg -d -T |grep -i Memory
[二 10月 5 13:21:05 2021 < 0.000000>] Base memory trampoline at [ffff99c040097000] 97000 size 24576
[二 10月 5 13:21:05 2021 < 0.000000>] Reserving 161MB of memory at 688MB for crashkernel (System RAM: 8110MB)
[二 10月 5 13:21:05 2021 < 0.000000>] Early memory node ranges
[二 10月 5 13:21:05 2021 < 0.000000>] Reserving Intel graphics memory at [mem 0xdd200000-0xdf1fffff]
[二 10月 5 13:21:05 2021 < 0.000000>] PM: Registered nosave memory: [mem 0x0009d000-0x0009dfff]
[二 10月 5 13:21:05 2021 < 0.000000>] PM: Registered nosave
…
```

- 6、实时监控查看日志末尾N行

```bash
#实时查看日志末尾10行
(base) [root@s186 ~]# watch “dmesg | tail -10”
```

- 7、查看指定级别格式日志

```bash
(base) [root@s186 ~]# dmesg -l warn
[ 0.000000] ACPI: RSDP 00000000000f0490 00024 (v02 DELL )
[ 0.000000] ACPI: XSDT 00000000d8fee080 00084 (v01 DELL CBX3 01072009 AMI 00010013)
[ 0.000000] ACPI: FACP 00000000d8ffa468 0010C (v05 DELL CBX3 01072009 AMI 00010013)
[ 0.000000] ACPI: DSDT 00000000d8fee198 0C2CA (v02 DELL CBX3 00000014 INTL 20091112)
[ 0.000000] ACPI: FACS 00000000da7fe080 00040
#支持的日志级别(优先级)：
emerg - 系统无法使用
alert - 操作必须立即执行
crit - 紧急条件
err - 错误条件
warn - 警告条件
notice - 正常但重要的条件
info - 信息
debug - 调试级别的消息
```

- 8、打印并清除内核环形缓冲区

```bash
(base) [root@s186 ~]# dmesg -c
…
(base) [root@s186 ~]# dmesg |more
(base) [root@s186 ~]#
```

- 9、直接查看dmesg日志信息
  
```bash
(base) [root@s186 log]# cat /var/log/dmesg |more
[ 0.000000] microcode: microcode updated early to revision 0x25, date = 2018-04-02
[ 0.000000] Initializing cgroup subsys cpuset
[ 0.000000] Initializing cgroup subsys cpu
[ 0.000000] Initializing cgroup subsys cpuacct
[ 0.000000] Linux version 3.10.0-957.5.1.el7.x86_64 (mockbuild@kbuilder.bsys.centos.org) (gcc version 4.8.5 20150623 (Red Hat 4.8.5-36) (GCC) ) #1
SMP Fri Feb 1 14:54:57 UTC 2019
[ 0.000000] Command line: BOOT_IMAGE=/vmlinuz-3.10.0-957.5.1.el7.x86_64 root=UUID=062b2cf4-b789-4eb4-bc1a-4b48d8266d16 ro crashkernel=auto biosdev
name=0 rhgb quiet LANG=en_US.UTF-8
```

## 三、使用语法及参数说明

- 1、使用语法

```bash
#dmesg [选项]
```

- 2、参数说明

| 参数选项 | 参数说明 |
| --- | --- |
| `-C, --clear` | 清除内核环形缓冲区(ring butter) |
| `-c, --read-clear` | 读取并清除所有消息 |
| `-D, --console-off` | 禁止向终端打印消息 |
| `-d, --show-delta` | 显示打印消息之间的时间差 |
| `-e, --reltime` | 以易读格式显示本地时间和时间差 |
| `-E, --console-on` | 启用向终端打印消息 |
| `-F, --file <文件>` | 用 文件 代替内核日志缓冲区 |
| `-f, --facility <列表>` | 将输出限制为定义的设施 |
| `-H, --human` | 易读格式输出 |
| `-k, --kernel` | 显示内核消息 |
| `-L, --color` | 显示彩色消息 |
| `-l, --level <列表>` | 限制输出级别 |
| `-n, --console-level <级别>` | 设置打印到终端的消息级别 |
| `-P, --nopager` | 不将输出通过管道传递给分页程序 |
| `-r, --raw` | 打印原生消息缓冲区 |
| `-S, --syslog` | 强制使用 syslog(2) 而非 /dev/kmsg |
| `-s, --buffer-size <大小>` | 查询内核环形缓冲区所用的缓冲区大小 |
| `-T, --ctime` | 显示易读的时间戳(如果您使用了SUSPEND/RESUME 则可能不准) |
| `-t, --notime` | 不打印消息时间戳 |
| `-u, --userspace` | 显示用户空间消息 |
| `-w, --follow` | 等待新消息 |
| `-x, --decode` | 将设施和级别解码为可读的字符串 |
| `-h, --help` | 显示此帮助并退出 |
| `-V, --version` | 输出版本信息并退出 |
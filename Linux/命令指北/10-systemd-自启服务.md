###### datetime:2025/03/26 14:41:00

###### author:nzb

# systemd详解

## 一、简介
`systemd` 是一种用于 `Linux` 操作系统的系统和服务管理器，被设计为 `init` 系统的替代品。它负责在系统启动时启动系统组件，以及在系统运行期间管理系统进程。自从 `Linux` 内核 2.6.32 版本后，许多主流的 `Linux` 发行版，如 `Fedora、Ubuntu、Debian` 和 `CentOS` ，都采用了 `systemd` 作为默认的初始化系统。

## 二、主要特点

- 并行启动服务：
  - `systemd` 能够并行地启动系统服务，缩短系统启动时间。
  - 通过服务的依赖关系，确保必要的服务按正确的顺序启动。
- 按需启动（ `Socket` 激活）：
  - 仅在需要时启动服务，减少资源消耗。
  - 使用套接字激活机制，`systemd` 可以在有请求时自动启动相关服务。
- 统一的进程管理：
  - 提供 `systemctl` 命令，统一管理系统服务的启动、停止、重启和状态查询。
  - 支持服务的自动重启、监控和日志记录。
- cgroups 资源控制：
  - 利用 `Linux` 内核的 `cgroups`（控制组）功能，`systemd` 可以限制和监控服务的资源使用，如 `CPU`、内存和 `I/O`。
- 日志系统（journald）：
  - 内置日志系统 `systemd-journald`，收集和管理系统日志。
  - 支持集中化的二进制日志存储，提供灵活的日志查询功能。
- 目标单元（Targets）：
  - 替代传统的运行级别（`runlevels`），提供更灵活的系统状态管理。
  - 通过目标单元，可以定义系统在不同状态下应该运行的服务集合。
- 定时器单元（Timers）：
  - 替代 `cron` 等传统定时任务调度器。
  - 可以为服务配置定时启动和周期性执行。

## 三、核心概念
- 单元（Units）：
  - systemd 以单元为基本管理对象，每个单元代表系统中的一个资源或服务。
  - 常见的单元类型：
    - **服务单元（.service）**：管理系统服务。
    - **目标单元（.target）**：表示系统的状态或运行级别。
    - **挂载单元（.mount）**：管理文件系统挂载点。
    - **套接字单元（.socket）**：用于套接字激活机制。
    - **定时器单元（.timer）**：管理定时任务。
    - **设备单元（.device）**：表示内核识别的设备。
    - **路径单元（.path）**：监控文件系统中的路径变化。

- 单元文件：
  - 定义单元的行为和配置，通常位于 `/lib/systemd/system/` 或 `/etc/systemd/system/` 目录下。
  - 单元文件的基本结构包括 `[Unit]`、`[Service]`、`[Install]` 等节。
- 依赖关系：
  - 单元之间可以通过依赖关系（如 `Requires`、`Wants`、`Before`、`After`）来定义启动顺序和条件。

## 四、常用命令

- 管理服务：

```shell
# 启动服务
 sudo systemctl start 服务名.service
 ​
 # 停止服务
 sudo systemctl stop 服务名.service
 ​
 # 重启服务
 sudo systemctl restart 服务名.service
 ​
 # 重新加载服务配置（不重启服务）
 sudo systemctl reload 服务名.service
 ​
 # 查看服务状态
 systemctl status 服务名.service
```

- 启用/禁用服务自启动：

```shell
 # 设置服务开机自启动
 sudo systemctl enable 服务名.service
 ​
 # 取消服务开机自启动
 sudo systemctl disable 服务名.service
 ​
 # 查看服务是否设置为开机自启动
 systemctl is-enabled 服务名.service
```

- 查看系统状态：

```shell
 # 查看系统所有已启动的服务和单元
 systemctl list-units
 ​
 # 查看所有可用的单元
 systemctl list-unit-files
​```    

- 日志查看：

```shell
 # 查看所有日志
 journalctl
 ​
 # 查看特定服务的日志
 journalctl -u 服务名.service
 ​
 # 实时查看日志（类似于 tail -f）
 journalctl -f
 ​
 # 查看系统最近一次启动的日志
 journalctl -b
```

## 五、自定义systemd服务

以下是创建自定义 systemd 服务的步骤：

### 编写服务文件：

在 `/etc/systemd/system/` 目录下创建一个新的服务文件，例如 `myservice.service`。

```text
 [Unit]
 Description=My Custom Service
 After=network.target
 ​
 [Service]
 Type=simple
 ExecStart=/usr/bin/mycommand --option
 Restart=on-failure
 RestartSec=5
 ​
 [Install]
 WantedBy=multi-user.target
```

- **[Unit]** 部分：
  - `Description`：对服务的简要描述。
  - `After`：指定服务的启动顺序。
- **[Service]** 部分：  
  - `Type`：服务类型，常用的有 `simple、forking、oneshot、notify` 等。
  - `ExecStart`：指定服务启动时执行的命令或脚本。
  - `Restart`：定义服务在退出时的重启策略，如 `no`、`on-success`、`on-failure`、`always` 等。
  - `RestartSec`：在服务重启前的等待时间。
- **[Install]** 部分：
  - `WantedBy`：定义服务的目标单元，通常设置为 multi-user.target，表示在多用户模式下启动。

### 重新加载 systemd 配置：

```shell
sudo systemctl daemon-reload
```

### 启动并启用服务：

```shell
 # 启动服务
 sudo systemctl start myservice.service
 ​
 # 设置服务开机自启动
 sudo systemctl enable myservice.service
```

### 查看服务状态和日志：

```shell
 # 查看服务状态
 systemctl status myservice.service
 ​
 # 查看服务日志
 journalctl -u myservice.service
```

## 六、示例

> 使用 `systemd` 管理 `kubectl port-forward`

可以将 `kubectl port-forward` 命令配置为 `systemd` 服务，以提高稳定性和可靠性。

### 创建服务文件：

创建文件 `/etc/systemd/system/thanos-port-forward.service`，内容如下：

```text
 [Unit]
 Description=Thanos Port Forward Service
 After=network.target
 ​
 [Service]
 Type=simple
 ExecStart=/usr/bin/kubectl port-forward --address 0.0.0.0 -n default svc/thanos-query-headless 8081:9090
 Restart=on-failure
 RestartSec=5
 StandardOutput=append:/data/forward/query.log
 StandardError=append:/data/forward/query.log
 ​
 [Install]
 WantedBy=multi-user.target
```

**注意：**
- `ExecStart` 中的命令需要使用命令的完整路径，可以通过 `which kubectl` 获取，例如 `/usr/bin/kubectl`。
- `StandardOutput` 和 `StandardError` 用于指定日志文件的位置，将服务的输出和错误追加到指定的日志文件中。

### 重新加载 systemd 配置：

```shell
sudo systemctl daemon-reload
```

### 启动并启用服务：

```shell
 # 启动服务
 sudo systemctl start thanos-port-forward.service
 ​
 # 设置服务开机自启动
 sudo systemctl enable thanos-port-forward.service
```

### 查看服务状态和日志：

```shell
 # 查看服务状态
 systemctl status thanos-port-forward.service
 ​
 # 查看服务日志
 journalctl -u thanos-port-forward.service
```

### 服务管理：

```shell
 # 停止服务
 sudo systemctl stop thanos-port-forward.service
 ​
 # 重启服务
 sudo systemctl restart thanos-port-forward.service
```

## 七、服务文件

在创建 `systemd` 服务文件时，需要了解各个配置项的含义。

- **[Unit]**
  - **Description**：对服务的描述。
  - **Documentation**：提供服务的文档链接。
  - **After**：定义服务的启动顺序，表示本服务在指定的服务或目标之后启动。
  - **Requires**：指定本服务依赖的其他服务，如果被依赖的服务无法启动，则本服务也无法启动。
  - **Wants**：类似于 `Requires`，但依赖关系较弱，如果被依赖的服务无法启动，`systemd` 仍会启动本服务。

- **[Service]**
  - **Type**：指定服务的启动类型，常见的有：
    - `simple`：默认类型，`ExecStart` 指定的命令为主进程。
    - `forking`：`ExecStart` 指定的命令会派生出一个子进程，父进程会退出，服务的主进程为子进程。
    - `oneshot`：适用于一次性运行的任务，`systemd` 会等待命令执行完成。
    - `notify`：服务启动后，会发送通知信号告知 `systemd`，适用于支持 `sd_notify` 的程序。
    - `idle`：服务会在其他任务执行完毕后再启动。
  - **ExecStart**：指定启动服务时执行的命令或脚本。
  - **ExecReload**：指定重新加载服务时执行的命令。
  - **ExecStop**：指定停止服务时执行的命令。
  - **Restart**：定义服务的重启策略：
    - `no`：不重启（默认）。
    - `on-success`：服务正常退出时重启。
    - `on-failure`：服务异常退出时重启。
    - `always`：无论退出状态如何，始终重启。
  - **RestartSec**：服务重启前的等待时间。
  - **User**：指定运行服务的用户。
  - **Group**：指定运行服务的用户组。
  - **Environment**：设置环境变量。
  - **WorkingDirectory**：指定工作目录。

- **[Install]**
    - **WantedBy**：指定服务所属的目标单元，当执行 `enable` 操作时，`systemd` 会在对应的目标单元目录下创建符号链接。
    - **RequiredBy**：类似于 `WantedBy` ，但依赖关系更强。

## 八、日志管理

### 使用 `journalctl` 查看日志：

```shell
 # 查看所有日志
 journalctl
 ​
 # 查看特定时间段的日志
 journalctl --since "2023-10-01" --until "2023-10-02"
 ​
 # 查看内核日志
 journalctl -k
 ​
 # 查看引导日志
 journalctl -b
 ​
 # 查看服务日志
 journalctl -u 服务名.service
 ​
 # 按照优先级过滤日志
 journalctl -p err
```

### 日志持久化配置：

默认情况下，`journald` 的日志存储在内存中，系统重启后日志会丢失。为了持久化日志，需要进行以下配置：

- 创建日志存储目录：

```shell
 sudo mkdir -p /var/log/journal
 sudo systemd-tmpfiles --create --prefix /var/log/journal
```

- 修改配置文件`/etc/systemd/journald.conf`，设置：

```text
 [Journal]
 Storage=persistent
 ```

- 重启`systemd-journald`服务：

```shell
sudo systemctl restart systemd-journald
```

## 九、定时器

`systemd` 提供了定时器单元，替代传统的 `cron` ，用于调度定时任务。

- 创建定时器服务单元：

创建服务文件 `/etc/systemd/system/mytask.service`：

```text
 [Unit]
 Description=My Scheduled Task
 ​
 [Service]
 Type=oneshot
 ExecStart=/usr/bin/mycommand --option
 ```

- 创建定时器单元：
  - **OnCalendar**：定义任务的调度时间。
  - **Persistent**：如果错过了预定时间，系统启动后立即执行。

创建定时器文件 `/etc/systemd/system/mytask.timer`：

```text
 [Unit]
 Description=Run MyTask every day at 2 AM
 ​
 [Timer]
 OnCalendar=*-*-* 02:00:00
 Persistent=true
 ​
 [Install]
 WantedBy=timers.target
```

- 启用并启动定时器：

```shell
 sudo systemctl daemon-reload
 sudo systemctl enable mytask.timer
 sudo systemctl start mytask.timer
```

- 查看定时器状态：

```shell
 systemctl list-timers
```

## 十、总结

`systemd` 是现代 `Linux` 系统中强大的系统和服务管理器，提供了统一的进程管理、日志记录和系统初始化功能。
通过 `systemd` ，可以更稳定和可靠地管理系统服务，如 `kubectl port-forward` 进程，避免因进程意外终止导致的服务不可用问题。
学习并掌握 `systemd` 的使用，对于系统运维和服务管理具有重要意义。

# 如何调整 systemd 加快启动

> 这些调整的真正好处不仅仅是缩短秒表的秒数。快速启动的系统会让人感觉响应更灵敏，从而减少等待时间。

## 识别启动过程中的慢点

**使用 `systemd-analyze` 跟踪启动性能**

在加速之前，您需要了解导致速度变慢的原因。`systemd-analyze` 命令可以显示内核和用户空间的初始化时间，让您大致了解启动时间。配合 `systemd-analyze blame` 命令使用，您将看到按启动时间排序的服务详细列表。真正的罪魁祸首通常就隐藏在这里，可能是配置错误的守护进程，也可能是您从未真正使用过的东西。

通过多次运行此分析，您可以了解一致性和异常值。某些服务可能会由于硬件检测而偶尔出现峰值，而其他服务则一直处于负载过重的状态。专注于最严重的问题，可以让您以最少的努力获得最大的改进。我通常会在进行更改之前保存一份输出副本，以便客观地衡量进度。

您还可以使用 `systemd-analyze critical-chain`，它显示了依赖项在启动顺序中的排列方式。那些阻碍其他重要任务的服务是重新排序或禁用的主要对象。有了这个工具，您就可以从猜测转向做出明智的调整，从而真正减少启动延迟。

## 减少后台服务

**禁用您实际上不使用的服务**

一旦知道哪些服务在浪费时间，下一步就是精简这些服务。 许多 Linux 发行版默认启用了一些并非每个用户都需要的服务。例如，即使在没有打印机或蓝牙硬件的机器上，打印机守护进程或蓝牙管理器也经常在后台运行。禁用这些服务可以节省启动过程中的宝贵时间。

最简单的管理方法是使用 `systemctl disable` 加上服务名称。这样可以阻止它在启动时启动，同时仍然允许您在需要时手动启动它。对于您绝对确定不会使用的服务， `systemctl mask` 会更进一步，完全阻止它们。systemd 需要启动的守护进程越少，您的机器就能越快达到可用状态。

> **注意**:不要随意禁用服务。务必仔细查看它们的作用以及哪些其他服务可能依赖于它们。在开始调整之前备份你的电脑绝对是个好主意。

当然，这需要谨慎一些。禁用一些关键功能可能会破坏您所依赖的功能，因此我建议您一次更改一个功能，并在每次调整后进行测试。几天后，您就可以构建一个更精简、更快速的启动配置文件，而不会破坏系统的稳定性。

## 利用 systemd 并行化

**优化依赖项以加快启动速度**

`systemd` 相较于旧版 `init` 系统的优势之一是它能够并行启动服务。它无需等待每个守护进程加载完毕再启动下一个，而是同时启动多个独立的守护进程。这意味着您的 CPU 和磁盘空间能够得到更高效的利用，从而自然而然地加快运行速度。关键在于确保正确定义依赖关系，避免服务之间不必要地相互阻塞。

您可以使用 `systemctl list-dependencies` 或直接查看单元文件来检查依赖关系。如果您发现某个服务正在等待它实际上并不需要的资源，则可以调整其配置。添加 `After=` 或 `Requires=` 等指令可以让您微调某个服务相对于其他服务的启动时间。移除不必要的依赖项可以避免空闲等待，并更好地利用并行化。

另一个技巧是为某些服务启用套接字激活。启用套接字激活后，`systemd` 仅在其套接字被访问时才启动守护进程，而不是每次启动时都启动。这不仅缩短了启动时间，还减少了后台资源的占用。正确调整后，您将获得一个速度更快、更轻量级的系统。

## 掩盖导致速度变慢的服务

**确保没有任何东西可以重新启用你屏蔽的内容**

有时仅仅禁用服务是不够的，因为其他软件包更新或依赖项可能会重新启用它。屏蔽是一个更强大的解决方案，因为它本质上将服务绑定到 `/dev/null` 这样它就不会被意外启动。这对于您确定与您的设置无关的服务尤其有用。一个很好的例子就是与您选择的管理器冲突的网络守护进程。

要屏蔽服务，请使用 `systemctl mask` 加上单元名称。此后，即使其他进程尝试启动该服务，`systemd` 也会拒绝。如果您改变主意，只需使用 `systemctl unmask` 即可轻松取消屏蔽。这样一来，您便可安心无虞，因为不需要的服务不会再次潜入您的启动序列。

问题在于，屏蔽错误的服务可能会造成混乱，尤其是在其他服务间接依赖该服务的情况下。因此，我总是在屏蔽之前仔细检查依赖关系树。但如果操作正确，屏蔽可以确保您的系统始终保持最佳状态 ，即使在更新过程中也是如此。

## 改进桌面会话的启动方式

**调整显示和登录管理器以提高速度**

对于桌面用户来说，启动过程的最后一步通常是图形会话。 像 GDM 、LightDM 或 SDDM 这样的显示管理器可能会增加各自的启动时间。调整它们或切换到更轻量级的显示管理器可以带来明显的效果。例如，LightDM 在中等硬件上通常比更重的显示管理器更快。

另一件需要注意的事情是会话设置中的自动启动应用程序。许多桌面环境会默认启动一些小型辅助应用、更新程序或云同步客户端。将这些程序精简到只启动您实际需要的程序，不仅可以加快桌面加载速度，还能减少登录后的混乱。这与禁用系统服务的原理相同，只是在用户级别应用。

## 为什么 systemd 调优在日常使用中会有回报

这些调整的真正好处不仅仅是缩短秒表的秒数。快速启动的系统感觉响应更快，并减少等待时间。通过分析、禁用、屏蔽和微调服务，您可以根据自己的需求打造更流畅的体验。如果出现问题，这些更改很容易撤消，但一旦调整到位，往往会持续有效。对我来说，这些小的进步每天都在积累，最终的回报就是一个感觉速度如预期般快速的 Linux 系统。
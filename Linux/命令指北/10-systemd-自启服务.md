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

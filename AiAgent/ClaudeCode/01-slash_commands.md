###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# Slash Commands

在交互式会话中，斜杠命令是控制 `Claude Code` 行为的最快捷方式。在任何提示符下输入 `/` 即可查看完整列表，或输入几个字母进行筛选。本模块涵盖您每天都会用到的内置命令。

## 发现命令

在提示符处输入 `/` ，即可显示包含所有可用命令的菜单。输入 `/` 可进行筛选——例如，输入 `/co` 可筛选出 `/compact` 、 `/color` 、 `/config` 、 `/context` 、 `/cost` 和 `/copy` 。使用方向键导航，回车键选择。当前设置中不可用的命令会自动隐藏，因此您只会看到有效的命令。

刚接触 `Claude Code`？试试 `/powerup` — 它会运行交互式课程，通过动画演示引导您直接在 `CLI` 中了解关键功能。

有些命令直接接受参数： `/compact focus on the API layer` ， `/model opus` ， `/effort high` ， `/rename auth-refactor` 。其他命令，如 `/context` ， `/cost` 和 `/status` 则不带任何参数立即运行。

## 命令类别

内置命令分为几个类别。了解这些类别有助于你快速找到正确的命令，而无需记住所有命令。

### Context management

> 上下文管理控制着`Claud Code`可以看到多少对话内容。

- `/context` — 显示上下文使用情况的可视化网格
- `/compact` — 压缩对话。传递指令以控制保留哪些内容： `/compact keep the migration plan, drop the debugging`
- `/clear` — 完全重新开始

### Session tools

> 会话工具帮助您管理会话。

- `/rename my-feature` — 为会话赋予一个易于理解的名称
- `/resume` — 继续之前的会话
- `/branch` — 创建一个平行对话，以便在不丢失当前状态的情况下探索替代方案。
- `/rewind` — 回滚到之前的某个时间点
- `/export` — 将会话保存到文件或剪贴板

### Configuration

> 配置命令可在会话过程中调整 `Claude` 的行为。

- `/model` — 在可用模型之间切换，例如 `Sonnet、Opus、Haiku` 以及其他别名，例如 `best` 或 `opusplan`
- `/effort` — 设置推理深度： `low` 、 `medium` 、 `high` 、 `xhigh` 、 `max （仅限会话）或 `auto`
- `/permissions` — 管理 `Claude` 无需请求即可执行的操作。
- `/config` — 打开设置菜单
- `/theme` (new in v2.1.118)— 用于创建和切换已命名的自定义主题。主题以 `JSON` 文件形式存储在 `~/.claude/themes/` 下，也可以直接手动编辑。

### Diagnostics

> 诊断功能有助于解决某些功能故障。

- `/cost` — 显示会话成本、持续时间、代码更改和令牌使用情况
- `/status` — 显示版本、型号和帐户信息
- `/doctor` — 检查安装健康状况
- `/diff` — 打开一个交互式查看器，用于查看未提交的更改，方便在提交之前查看 `Claude` 所做的更改。










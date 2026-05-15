###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# 命令详解

## 上下文和会话管理

每个 `Claude Code` 会话都有一个上下文窗口。`/context` 会将其可视化为一个彩色网格————绿色表示可用，黄色表示即将满，红色表示几乎耗尽。当上下文内容过长时， `/compact` 命令会压缩对话。传递焦点指令可以保留重要信息： `/compact focus on the database migration plan` 。

`/branch` 会从当前位置创建一个平行对话，让你可以并排探索两种不同的方法。`/rewind` 会回滚到之前的某个时间点————这在 `Claude` 走错路时非常有用。它还可以选择撤销文件更改，相当于撤销对话和代码的操作。

会话恢复功能使长时间工作成为可能。`/rename my-feature` 会将当前会话保存为一个易于理解的名称。`/resume my-feature` 会在稍后恢复会话，并保留所有上下文信息。使用 `/export` 可以将会话导出到文件或剪贴板，以便共享或存档。

```bash
/context
/compact focus on the auth refactor
/branch
/rename auth-refactor-v2
/export auth-refactor-v2.md
```

## 捆绑技能

`Claude Code` 内置了一些类似命令的技能，无需安装即可随时使用。

`/simplify` 命令会审查最近更改过的文件，检查代码质量，并生成多个并行审查代理来关注不同的问题。`/batch <instruction>` 命令用于处理跨多个文件的大规模更改————它会规划工作，使用隔离的 `Git` 工作树，并可以协调验证和面向 `PR` 的后续工作。`/loop 5m check deploy status` ，这对于轮询长时间运行的操作非常有用。`/proactive` 是 `/loop` 的别名————行为相同，但当目标是“持续观察并根据观察到的情况采取行动”而不是“定时运行此命令”时，它通常更易读。

`/debug` 启用详细日志记录，以帮助诊断 `Claude` 的行为或工具使用方面的问题。`/claude-api` 加载项目语言的 `Anthropic SDK` 参考————当检测到从 `@anthropic-ai/sdk` 或 `Python anthropic` 包导入的内容时，它会自动激活。


```bash
/simplify
/batch add JSDoc comments to all public functions in src/
/loop 2m check if the build finished
/debug
```

## 快速模式

快速模式是 `Opus 4.6` 的一种高速 `API` 配置，它优先考虑速度而非成本效益——速度大约提升 2.5 倍，但令牌成本也更高。它并非一个独立的模型，而是同一个 `Opus 4.6` 版本，只是配置有所不同。可以通过 `/fast` 启用，也可以在用户设置中将 `fastMode: true` 。启用后，提示栏旁边会出现一个 `↯` 图标。

```bash
/fast          # toggle on/off
/fast on       # explicitly enable
/fast off      # explicitly disable
```

快速模式会在您当前使用其他型号时自动切换到 `Opus 4.6`。关闭快速模式后，您将保持在 `Opus 4.6`————使用 `/model` 切换型号。`Opus 4.7` 可通过选择器以 `opus-4-7` 或 `claude-opus-4-7` 形式提供。

快速模式和努力程度是两个独立的速度控制选项。`/fast` 可以降低延迟 `/fast` 但不会影响质量。`/effort low` 可以减少思考时间，但这可能会降低复杂任务的质量。对于简单的任务，可以同时使用这两个选项以获得最高速度：

```bash
/fast
/effort low
```

当达到快速模式速率限制时，它会自动回落到标准的 `Opus 4.6` 速度（ `↯` 图标变为灰色），并在冷却时间结束后重新启用。快速模式需要在您的帐户上启用额外的流量，并且在基岩版、`Vertex AI` 或 `Foundry` 版本中不可用。企业管理员可以通过管理设置来控制其可用性。

## 键盘快捷键和电源功能

`Shift+Tab` 可在权限模式之间切换。顺序为： `default` 、 `acceptEdits` 、 `plan` ，以及可选模式（例如 `auto` 或 `bypassPermissions` 前提是您的环境中已启用这些模式）。这是在执行复杂任务时快速切换到计划模式，并在任务完成后切换回默认模式的方法。

`Option+T` （`macOS`）或 `Alt+T` 可切换至扩展思考模式—— `Claude` 会在做出回应前花更多时间进行推理。使用 `/effort` 设置推理深度： `auto` 、 `low` 、 `medium` 、 `high` 、 `xhigh` 或 `max` 适用）。`max` 仅对当前会话有效。`Ctrl+O` 进入详细模式，可实时查看工具调用和思考步骤。

`/btw your question` 提出了一个旁支问题，但并未将其添加到对话历史记录中————这对于核实事实或询问语法问题非常有用，而不会使上下文变得混乱。`Ctrl+B` 可以将正在运行的 `bash` 命令和代理程序置于后台，这样你就可以在它们继续工作的同时向 `Claude` 发送其他指令。如果需要终止所有后台代理程序，官方快捷键是 `Ctrl+X` `Ctrl+K` 。

`Ctrl+U` 会清除整个输入缓冲区， `Ctrl+Y` 会恢复刚刚清除的内容————当你输入了一长串提示符并想重新开始而不丢失之前的内容时，这非常有用。`Ctrl+L` 除了清除提示符输入外，还会强制全屏重绘，这在终端输出出现撕裂或漂移时很有用。在转录查看器底部， `[` 会将转录内容滚动到回滚窗口， `v` 会在你的 `$EDITOR` 中打开它。

`/diff` 命令会打开一个交互式差异查看器，用于查看未提交的更改——这比阅读原始 `Git` 输出要好得多，尤其是在你想在提交之前查看 `Claude` 做了什么的时候。`/insights` 会生成一份会话分析报告，其中包含有关已完成工作的统计信息。

```bash
# Toggle to plan mode, then back
Shift+Tab
Shift+Tab

/effort high
/btw what's the difference between async and defer on script tags?
```

## 可视化模式

`Vim` 用户可以在输入编辑器中使用可视选择功能。按 `v` 进行字符选择，按 `V` 进行行选择。进入可视模式后，可以使用导航键（ `h` 、 `j` 、 `k` 、 `l` 、 `w` 、 `e` 、 `b` 、 `f` 、 `F` 、 `t` 、 `T` ）扩展选择范围。然后应用运算符：

`d / x` 删除， `y` 复制， `c / s` 更改， `p` 替换为寄存器内容， `r{char}` 替换每个选定的字符， `~ / u / U` 切换或强制大小写， `> / <` 缩进或取消缩进， `J` 连接行， `o` 交换光标和锚点。文本对象如 `iw` 、 `aw` 、 `i"` 、 `a"` 、 `i(` 、 `a(` 可用于精确选择。不支持块级可视模式（ `Ctrl+V` ）。

## 统一会话统计

`/usage` 命令会显示一个统一的仪表盘，它整合了之前 `/cost` 和 `/stats` 分别显示的内容————包括总费用估算、`API` 和实际使用时长，以及新增或移除的线路。对于 `API` 用户 `/cost` 它包含详细的令牌统计信息；对于订阅用户，它显示套餐使用情况条和活动记录。`/cost` 和 `/stats` 仍然作为快捷方式，可以打开相应的标签页。美元金额为本地计算的估算值——请查看 `Claude` 控制台以获取权威账单信息。

## 目标导向型课程

`/goal` 命令会设定一个完成目标， `Claude` 会在多个回合中努力达成该目标。目标激活后，`Claude` 会持续自主行动，直到达成目标为止。实时叠加层会显示已用时间、回合数和代币使用情况，方便您无需中断游戏即可监控进度。

```bash
/goal migrate all API endpoints from REST to GraphQL
/goal all tests pass and coverage is above 80%
```

对于复杂的多步骤任务，目标与 `/effort high` 指令搭配使用效果更佳。要限制 `Claude` 执行的回合数，请将 `CLAUDE_CODE_MAX_TURNS` 设置为环境变量：

```bash
export CLAUDE_CODE_MAX_TURNS=50
```

## 自定义主题

`/theme` 命令允许您创建和切换已命名的颜色主题。主题以 `JSON` 文件的形式存储在 `~/.claude/themes/` 下，并且可以手动编辑。插件也可以通过插件包中的 `themes/` 目录来分发主题。

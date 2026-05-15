###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# Agent Skills

技能是 `Claude` 根据上下文自动发现和使用的可重用功能。它们比简单的命令更强大：支持渐进式加载以保持轻量级、动态 `shell` 上下文注入、子代理隔离和调用控制。本模块将向您展示如何设计和构建高效的技能。

## 如何加载技能

`Claude` 尽量简化技能加载。技能描述会被加载，以便 `Claude` 了解可用的能力。完整的 `SKILL.md` 文件内容仅在技能被调用时才会加载，而辅助文件也仅在需要时才读取。

这意味着你可以安装很多技能而不会让上下文窗口显得杂乱无章。`Claude` 会根据技能描述识别它们，然后只加载它决定使用的技能的实际指令。

项目范围内的技能位于 `.claude/skills/<name>/SKILL.md` （已提交到 `Git`），个人范围内的技能位于 `~/.claude/skills/<name>/SKILL.md` 。插件技能使用 `plugin-name:skill-name` 命名空间，因此不会与项目或个人技能冲突。当非插件技能名称相同时，优先级顺序为：企业级技能 < 个人级技能 < 项目级技能。

`Claude` 还能自动发现项目根目录下子目录中嵌套的 `.claude/skills/` 目录中的技能。例如，如果您在 `packages/frontend/` 目录下工作，`Claude` 也会找到 `packages/frontend/.claude/skills/` 中定义的技能。这使得在单体仓库架构中，技能可以轻松地与特定的包或服务放在一起。

```bash
.claude/skills/code-review/
├── SKILL.md              # Instructions (required)
├── templates/
│   └── review-checklist.md
└── scripts/
    └── analyze-metrics.py
```

## 如何撰写有效的技能描述

描述字段是技能中最重要的部分。它控制着 `Claude` 何时自动调用该技能，并且必须包含足够的信息，以便 `Claude` 将其与真实用户的请求匹配。像“帮助编写代码”这样模糊的描述永远不会触发技能。而包含具体触发词的描述则有效：

```md
---
name: security-review
description: Scan code for security vulnerabilities including injection flaws, authentication issues, and data exposure. Use when reviewing code changes, preparing a PR, or when the user mentions security, vulnerabilities, or audit.
---
```

请包含任务类型（“扫描”、“生成”、“分析”）、主题领域（“安全”、“API”、“数据库”）和明确的触发短语（“当用户提及时”、“使用时机”）。技能列表会将每个条目的 `description` 加上 `when_to_use` 文本的总长度限制为 1536 个字符，因此请预先填写关键用例，并将超出限制的触发短语添加到 `when_to_use` 中。

```md
---
name: security-review
description: Scan code for security vulnerabilities including injection flaws, authentication issues, and data exposure.
when_to_use: When reviewing code changes, preparing a PR, or when the user mentions security, vulnerabilities, or audit.
---
```

`Claude` 将技能描述的总空间预算为上下文窗口的 `1%` 左右，必要时会回退到 `8000` 个字符，而 `SLASH_COMMAND_TOOL_CHAR_BUDGET` 可以让你提高这个上限。运行 `/context` 可以检查技能是否被排除在列表之外。

辅助文件扩展了技能功能，但不会扩展二级上下文。请在 SKILL.md 中使用相对路径引用它们：

```md
For the full review checklist, see [templates/review-checklist.md](templates/review-checklist.md).
```

`Claude` 会在需要时使用 `bash` 读取支持文件。请将 `SKILL.md` 控制在 `500` 行以内；详细的参考资料请放在单独的文件中。

## 动态上下文和调用控制

- `!command` 语法会在技能内容到达 `Claude` 之前执行 `shell` 命令。输出会被内联———— `Claude` 只能看到结果，看不到命令​​本身。这就是如何为技能提供实时上下文：

    ```md
    ---
    name: pr-summary
    description: Summarize pull request changes. Use when asked to review or summarize a PR.
    context: fork
    agent: Explore
    ---

    ## PR context
    - Diff: !`gh pr diff`
    - Comments: !`gh pr view --comments`
    - Changed files: !`gh pr diff --name-only`

    Summarize the intent and key changes in this pull request.
    ```

- `shell` 字段指定用于 `!command` 块的 `shell` 。在 `Windows` 上启用 `PowerShell` 工具时（ `CLAUDE_CODE_USE_POWERSHELL_TOOL=1` ），请将其设置为 `powershell` 而不是默认的 `bash` ：

    ```md
    ---
    name: windows-helper
    description: Manage Windows services and configurations
    shell: powershell
    ---
    ```

    两个 `frontmatter` 字段控制谁可以调用技能。`disable disable-model-invocation: true` 表示只有用户才能通过 `/skill-name` 命令调用技能———— `Claude` 永远不会自动触发它。对于任何带有副作用（部署、推送、发送）的技能，请使用此选项。`user user-invocable: false` 会将技能从 `/` 菜单中隐藏，但仍然允许 `Claude` 自动调用它————这适用于无法作为命令执行的背景知识技能。

- `paths:` 接受一个 `YAML` 格式的 `glob` 列表，用于指定技能的适用范围。设置后，技能仅在工作目录与其中一个 `glob` 匹配时加载。这样可以防止项目特定的技能污染无关的会话。
- `effort` 控制着该技能的推理深度。数值包括 `low` 、 `medium` 、 `high` 、 `xhigh` 和 `max` （仅限会话）。 `low` 适用于快速查找或生成样板代码， `medium` 适用于大多数任务， `high` 适用于需要仔细推理的深度分析：

    ```md
    ---
    name: security-review
    description: Scan code for security vulnerabilities.
    effort: high
    ---
    ```

- `context`: `fork` 会在一个独立的子代理中运行该技能，该子代理拥有自己的上下文窗口。 
- `agent` 字段指定代理类型： `Explore` 用于只读研究， `Plan` 用于规划， `general-purpose` 用于任何需要所有工具的情况。主对话保持简洁，而子代理则负责繁重的工作。
- `model` 字段指定技能激活时要使用的模型。当任务受益于特定模型的优势时，这非常有用（例如， `opus` 用于复杂推理， `sonnet` 用于快速执行）：

    ```md
    ---
    name: deep-analysis
    description: Thoroughly analyze the codebase for a specific pattern or issue
    context: fork
    agent: Explore
    model: opus
    disable-model-invocation: true
    ---

    Analyze $ARGUMENTS across the entire codebase:
    1. Use Glob and Grep to find all occurrences
    2. Read each file and understand context
    3. Summarize patterns, inconsistencies, and recommendations
    ```

## 参数和工具访问

技能接受两种参数方式。`$ARGUMENTS` 会将命令名称之后的所有内容作为一个字符串捕获。`$0` 、 `$1` 、 `$2` 则会捕获以空格分隔的各个参数。这两种方式都会在提示符到达 `Claude` 之前进行替换。`argument-hint` 通过显示技能所需的参数来改进斜杠菜单的自动完成功能：

```md
---
name: review-pr
description: Review a GitHub PR by number
argument-hint: "<pr-number> <priority>"
allowed-tools: Bash(gh *), Read, Grep, Glob
---

Review PR #$0 with priority $1. Focus on security and performance.

Reference our standards in [standards/code-review.md](standards/code-review.md).
```

- 用法： `/review-pr 456 high` 
  - `$0` 变为 `456` 
  - `$1` 变为 `high` 。

- `allowed-tools` 限制技能运行时可以使用的工具，其语法模式与权限规则相同。这对于只能读取或只能与特定命令行工具交互的技能非常有用。
- `.claude/commands/*.md` 目录下的旧版命令文件仍然有效，但建议使用技能文件格式。如果两种文件同名，则技能文件优先。

## 内置技能

`Claude Code` 内置了一些无需安装即可随时使用的技能。`/fewer-permission-prompts`（v2.1.112 版本新增）会扫描您的对话记录 `/fewer-permission-prompts` 查找常见的只读 `Bash` 和 `MCP` 工具调用，然后为您的 `.claude/settings.json` 文件生成一个优先级允许列表。在几次会话后运行此命令，即可生成一个符合您实际工作流程的权限配置

```bash
/fewer-permission-prompts
```
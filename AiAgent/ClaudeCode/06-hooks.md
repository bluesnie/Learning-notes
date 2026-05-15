###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# Hooks

钩子是当 `Claude Code` 会话期间发生特定事件时自动执行的脚本。它们通过标准输入 (`stdin`) 接收 `JSON` 输入，并通过退出代码和 `JSON` 输出传递结果。命令钩子是确定性的、可组合的、可测试的，并且与编程语言无关。提示钩子和代理钩子使用 `Claude` 模型进行评估，因此它们的行为是不确定的。本模块将介绍钩子系统、关键事件以及如何编写有用的钩子。

## 钩子架构和配置

钩子在配置文件中以 `hooks` 键进行配置。每个事件都有一个匹配器数组，每个匹配器又包含一个钩子定义数组。 `matcher` 字段是一个正则表达式模式，用于匹配工具名称

- "`Bash`" 匹配完全匹配
- "`Write|Edit`" 匹配两者之一
- "`*`" 匹配所有工具
- "`mcp__github__.*`" 匹配所有 `GitHub MCP` 工具。

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/validate-bash.py\"",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

匹配器还支持条件 `if` 字段（`v2.1.85`），该字段使用权限规则语法来进一步筛选钩子触发的时机。 `matcher` 按名称选择工具，而 `if` 字段则缩小范围，仅针对该工具的特定调用。当您只关心工具调用的子集时，这非常有用————例如，拦截 `git push` 命令，而无需在每次 `Bash` 调用时都运行它：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "if": "Bash(git push*)",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/check-push.sh"
          }
        ]
      }
    ]
  }
}
```

`Claude Code` 支持 `30` 多个钩子事件。日常工作中最常用的事件包括 `PreToolUse` （在工具运行前进行验证，可以阻塞）、 `PostToolUse` （在工具运行后进行观察或响应，可以添加上下文）、 `UserPromptSubmit` （在 `Claude` 处理用户输入前拦截输入）和 `Stop` （在 `Claude` 完成响应后运行检查）。此外，还有用于权限处理（ `PermissionRequest` ）、通知、子代理生命周期（ `SubagentStart` 、 `SubagentStop` ）、故障（ `PostToolUseFailure` 、 `StopFailure` ）、配置更改、文件监视（ `FileChanged` ）、上下文压缩（ `PreCompact` 、 `PostCompact` ）和工作树管理的事件。

一些新增事件扩展了钩子可以响应的事件类型。`CwdChanged` 在工作目录更改时触发，支持类似 `direnv` 的响应式环境管理——例如，当 `Claude` 进入项目目录时自动加载环境变量。`TaskCreated` 在使用 `TaskCreate` 工具时触发，因此您可以在任务创建时记录或验证新任务。`WorktreeCreate` 在创建工作树代理时触发，并支持 `type: "http"` 进行远程通知——这在并行工作开始时提醒外部 `Elicitation` 非常有用。`Elicitation` 在 `MCP` 服务器通过交互式对话框在任务执行过程中请求结构化用户输入时触发，并且可以在向用户显示请求之前拦截并修改该请求。`ElicitationResult` 在用户响应 `MCP` 请求后触发，并且可以在响应发送回 `MCP` 服务器之前拦截并覆盖该响应。

`PreCompact` 会在 `Claude Code` 将对话压缩以释放上下文之前运行，它可以阻止压缩操作————这在您想要快照状态、警告用户或否决会丢弃关键上下文的自动压缩时非常有用。事件 `matcher` 会区分触发原因： "`manual`" 表示用户运行了 `/compact` 命令， "`auto`" 表示由于上下文已满，`Claude Code` 自动进行了压缩。使用退出代码 `2` 或 `JSON` 决策有效负载阻止此操作：

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "auto",
        "hooks": [
          { "type": "command", "command": "./scripts/snapshot-context.sh" }
        ]
      }
    ]
  }
}
```

从脚本返回 `{"decision": "block", "reason": "active refactor in flight"}` 保持对话不变。`PostCompact` 在压缩成功后触发，此时可以重新附加注释、重新调用技能或记录已保存的内容。

`Hook` 脚本通过标准接收 `JSON` 数据，并可访问 `Claude Code` `自动设置的多个环境变量。CLAUDE_CODE_SESSION_ID` 包含唯一的会话标识符——可将其用于将 `Hook` 日志和外部遥测数据与特定会话关联起来。

`Python` 钩子函数会像这样读取 `JSON` 输入：

```json
import json, sys, os
data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})
session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
```

退出代码 `0` 表示成功（解析 `JSON` 标准输出以获取输出）。退出代码 2 表示阻塞性错误———— `Claude` 停止运行并显示标准错误输出信息。任何其他退出代码均为非阻塞性警告，以详细模式显示。

钩子输入包含一个带有当前工作量级别的 `effort` 对象： { "`effort`": { "`level`": "`medium`" } } 。可用级别包括 `low` 、 `medium` 、 `high` 、 `max` 和 `auto` 。该值也可作为钩子脚本中的 `$CLAUDE_EFFORT` 环境变量使用，钩子运行的 `Bash` 工具命令也可以读取该值：

```json
import json, os, sys
data = json.load(sys.stdin)
effort_level = data.get("effort", {}).get("level", "medium")  # from JSON
effort_env = os.environ.get("CLAUDE_EFFORT", "medium")          # from env var
```

## 常见钩子类型和样式

命令钩子支持两种形式。 

- `Shell` 形式 （默认）将 `command` 字符串传递给 `shell` 进行标记化。 
- `Exec` 形式会在 `command` 旁边设置一个 `args` 数组，直接启动进程而无需 `shell` ————这避免了 `shell` 转义问题，并且对于带有用户提供参数的命令来说更加安全：

```json
{
  "type": "command",
  "command": "node",
  "args": ["./scripts/validate.js", "--strict"]
}
```

钩子可以通过五种方式运行。

- 命令(`command`)钩子执行本地 `shell` 命令
- 提示符(`prompt`)钩子要求 `Claude` 执行提示符，通常是在 `Stop` 或 `SubagentStop` 事件发生时。
- `agent` 钩子生成子代理以进行多步骤验证
- `HTTP` 钩子将相同的 `JSON` 有效负载 `POST` 到 `webhook` 端点，这对于远程日志记录或策略服务非常有用。`HTTP` 钩子支持在请求头中插入环境变量，
- `mcp_tool` 钩子直接调用 `MCP` 工具————当钩子需要调用外部服务（例如发布到 `Slack` 或创建 `GitHub issue`）而无需调用外部服务时非常有用。注意：应用内配置构建器目前尚不支持生成 `mcp_tool` 钩子——请参考以下 `JSON` 示例：


```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "mcp_tool",
            "server": "slack",
            "tool": "send_message",
            "input": { "channel": "#deploys", "text": "Claude finished the task" }
          }
        ]
      }
    ]
  }
}
```

`PostToolUse` 和 `PostToolUseFailure` 钩子输入包含一个 `duration_ms` 字段，用于记录工具的执行时间（以毫秒为单位，不包括权限提示和 `PreToolUse` 钩子）。您可以使用此字段跟踪运行缓慢的工具，或在工具运行时间超过阈值时设置警报。

## 常见钩子模式

文件保存时自动格式化是最有用的钩子之一。 `Write` `Write|Edit` 上的 `PostToolUse` 钩子会自动运行格式化程序，因此 `Claude` 的输出始终是干净的

```bash
#!/bin/bash
INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))")
case "$FILE" in
  *.ts|*.tsx|*.js) prettier --write "$FILE" 2>/dev/null ;;
  *.py) black "$FILE" 2>/dev/null ;;
  *.go) gofmt -w "$FILE" 2>/dev/null ;;
esac
exit 0
```

写入操作的安全扫描使用 `PostToolUse` 和 `additionalContext` 输出，以警告 `Claude` 它刚刚写入的潜在秘密信息：

```python
SECRET_PATTERNS = [
    (r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", "Potential hardcoded API key"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "Potential hardcoded password"),
]
# ... check content, then:
output = {"hookSpecificOutput": {"hookEventName": "PostToolUse",
  "additionalContext": f"Security warnings: {'; '.join(warnings)}"}}
print(json.dumps(output))
```

阻止危险命令使用 `PreToolUse` ，并进行正则表达式检查，退出代码为 `2`：

```python
BLOCKED = [(r"\brm\s+-rf\s+/", "Blocking dangerous rm -rf /")]
for pattern, message in BLOCKED:
    if re.search(pattern, command):
        print(message, file=sys.stderr)
        sys.exit(2)
```

## 高级：提示钩子和组件范围

对于 `Stop` 和 `SubagentStop` 事件，钩子类型 "`prompt`" 使用 `LLM` 来评估任务完成情况。 `LLM` 读取对话并返回一个结构化的决策，指示是否允许 `Claude` 停止或继续工作。这对于具有明确完成标准的任务非常有效：

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Check: 1) Were all files modified? 2) Do tests pass? 3) Is the PR description updated? If anything is missing, explain what.",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

"`agent`" 类型的钩子会生成一个子代理来执行评估——与提示钩子（单轮）不同， `agent` 钩子可以使用工具并执行多步骤推理。当检查需要读取文件或运行命令时，请使用此钩子。

还可以使用 `hooks` `frontmatter` 字段将 `hooks` 的作用域限定于特定的技能和代理。技能 `frontmatter` 中的 `PreToolUse hook` 仅在该技能执行期间触发：

```markdown
---
name: production-deploy
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/production-safety-check.sh"
          once: true
---
```

`once`: `true` 标志仅使钩子函数在每个会话中运行一次，而不是在每次使用匹配的工具时都运行。这对于只需执行一次的设置检查非常有用。


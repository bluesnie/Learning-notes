###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# Model Context Protocol (MCP)

`MCP`（模型上下文协议）使 `Claude` 能够实时访问外部服务。与存储静态上下文的内存文件不同，`MCP` 连接允许 `Claude` 查询实时数据——例如您的 `GitHub` 问题、生产数据库、`Slack` 频道或任何具有 `MCP` 服务器的服务。本模块涵盖添加服务器、理解作用域以及如何有效地使用 `MCP` 工具。

## 添加 MCP 服务器

添加服务器最快的方法是使用 `claude mcp add` 命令。选择与服务器类型匹配的传输方式：远程服务器使用 `http` ，本地运行的进程使用 `stdio` ，尚未迁移到 `HTTP` 的旧式远程服务器使用 `sse` 。 （注意：`SSE` 已弃用——请尽可能使用 `HTTP` 服务器。） 在原生 `Windows` 系统中，启动基于 `npx` 的 `stdio` 服务器时，通常会使用 `cmd /c` 命令。

```bash
# Add a remote HTTP server
claude mcp add --transport http notion https://mcp.notion.com/mcp

# Add a local Node.js server via stdio
claude mcp add --transport stdio github -- npx @modelcontextprotocol/server-github

# Add with an auth header
claude mcp add --transport http my-api https://api.example.com/mcp \
  --header "Authorization: Bearer $MY_TOKEN"
```

使用 `claude mcp list` 、 `claude mcp get <name>` 和 `claude mcp remove <name>` 命令管理您的服务器。在会话中使用 `/mcp` 命令可以显示活动连接，并为需要基于浏览器的身份验证的服务器触发 `OAuth` 流程。其他有用的命令包括 `claude mcp reset-project-choices` 、 `claude mcp add-from-claude-desktop` 和 `claude mcp serve` （当您希望 `Claude Code` 本身充当 `MCP` 服务器时）。

`MCP` 配置位于 `~/.claude.json` （您的本地用户配置文件）或项目根目录下的 `.mcp.json` （与团队共享）。. `.mcp.json` 文件已提交到 `Git` ，并在首次使用时提示团队成员进行批准。环境变量扩展适用于所有配置字段——使用 `${VAR:-default}` 可设置备用方案：

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

`MCP` 服务器现在默认并发连接。当您配置了多个服务器（包括本地标准 `I/O` 服务器和远程 `claude.ai` 连接器）时，它们会在启动时并行初始化，而不是逐个初始化。这显著降低了包含多个 `MCP` 集成的项目的启动延迟。

使用 `MCP_CONNECT_TIMEOUT_MS` 设置每个服务器的连接超时时间（以毫秒为单位）（默认值因传输方式而异）：

在 `Claude.ai` 中配置的 `MCP` 连接器也可以自动出现在 `Claude Code` 中。如果您通过 `Web` 界面设置了服务器，则无需单独的本地配置，该服务器即可在您的 `CLI` 会话中使用。如果同一服务器同时在本地和通过 `Claude.ai` 配置，则会自动去重，从而避免对同一服务建立两个连接。

## 作用域和工具发现

`MCP` 配置有三个作用域。本地作用域（存储在项目密钥下的 `~/.claude.json` 文件中）是私有的，仅对您和当前项目有效。项目作用域（ `.mcp.json` 文件）通过 `Git` 与团队共享。用户作用域（全局 `~/.claude.json` 文件）适用于您的所有项目。

当同一服务器在多个作用域中定义时，本地配置优先。这样，您就可以使用本地版本覆盖团队范围的服务器配置进行测试，而不会影响其他任何人。

`MCP` 提示符以斜杠命令的形式出现，格式 `/mcp__servername__promptname` 资源可以通过 `@server:protocol://resource/path` 内联引用。工具搜索默认启用———— `MCP` 工具会延迟加载并在需要时发现，因此只有 `Claude` 实际用于任务的工具才会进入上下文。每个 `MCP` 工具的描述和服务器指令的大小都限制在 `2KB` 以内，以防止 `OpenAPI` 生成的服务器占用过多上下文空间。当 `MCP` 工具的输出超过上下文窗口的 `10%` 时，会显示运行时警告。

要覆盖特定服务器的延迟加载，请将 `alwaysLoad: true` 添加到其配置中 — 来自该服务器的所有工具都将跳过工具搜索延迟加载，并始终在会话中可用：

子代理范围的 `MCP` 允许您授予特定代理对会话其余部分不需要的服务器的访问权限：

```markdown
---
name: data-analyst
description: Analyze production data
mcpServers:
  - database
  - playwright:
      type: stdio
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
---
```

## 实用使用模式

连接 `GitHub MCP` 后，您可以使用自然语言处理 `PR` 、问题和提交。`Claude` 会查询服务器，获取实时数据并做出响应：

```markdown
List all open PRs that haven't been reviewed in more than 3 days.
Create an issue for the login timeout bug with medium priority.
/mcp__github__pr_review 456
```

数据库 `MCP` 支持自然语言查询，无需自行编写 `SQL` 代码：

```markdown
Find all users who placed more than 5 orders in the last 30 days.
What's the average order value by country for Q1 2026?
```

对于复杂的流程，多个 `MCP` 服务器可以自然地协同工作。例如，一个每日报告流程可能包括：从 `GitHub MCP` 获取 `PR` 指标，从数据库 `MCP` 查询销售数据，使用文件系统 `MCP` 生成报告，并通过 `Slack MCP` 发布——所有操作都可以在单个会话中完成。

`MCP` 请求允许服务器暂停工作​​流并向用户请求结构化输入。当服务器需要无法自行获取的信息时——例如 `OAuth` 授权、执行破坏性操作前的确认，或包含项目特定参数的表单——它会触发一个交互式对话框。用户可以看到表单字段或浏览器 `URL` ，提供响应后，服务器会从中断处继续执行。 `Elicitation` 和 `ElicitationResult` 钩子允许您以编程方式拦截或自定义这些对话框。

安全最佳实践：始终使用环境变量存储凭据，切勿将令牌提交到 `Git` ，仅在需要查询数据时使用只读令牌，并将服务器访问权限限制在必要的最小范围内。对于企业部署， `managed-mcp.json` 允许管理员在组织范围内强制执行允许访问的服务器列表。

其他值得了解的重要 `MCP` 功能： `MCP` 服务器可以发送 `list_changed` 通知，动态更新其可用工具、提示和资源，而无需重新连接。如果 `HTTP` 或 `SSE` 服务器在会话期间断开连接，`Claude Code` 会自动使用指数退避算法重新连接——最多尝试五次，每次延迟一秒，之后每次延迟翻倍。 `MCP` 服务器还可以通过 `claude/channel` 功能将消息推送到您的会话中，使 `Claude` 能够对 `CI` 结果、监控警报或聊天消息等外部事件做出反应。

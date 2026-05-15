###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# Setting Up Claude Code for a Project

让 `Claude Code` 在项目中正常运行只需大约十分钟的设置时间。好处是，`Claude` 从第一条消息开始就能理解你的约定，拥有执行有效工作的必要权限，并且对团队中的每个人都保持一致的行为。本模块将按顺序介绍设置步骤。

## 初始化项目内存

首先执行 `/init` 。`Claude` 会扫描你的代码库——读取 `package.json` 、现有文档和目录结构——并生成一个 `CLAUDE.md` ，其中包含你的技术栈、关键命令和初始约定。请立即将此文件提交到 `Git`，以便团队成员获得相同的上下文信息。

一份优秀的 `CLAUDE.md` 文件应该简洁明了、重点突出。每个文件最好控制在 200 行以内。每一行都应该与几乎所有会话相关——如果某项内容仅对某个功能重要，则应将其放在一个路径作用域的规则文件中。最有价值的部分包括：技术栈和版本、开发命令（安装、测试、构建、代码检查）、不常见的命名约定，以及容易让新手开发者犯错的已知陷阱。

```text
# Project: Payment Service

## Stack

- Node.js 20, TypeScript 5, PostgreSQL 15
- Express for API, Prisma for ORM, Jest for tests

## Commands

- `npm run dev` — start with hot reload
- `npm test` — run test suite
- `npm run migrate` — apply pending migrations
- `npm run lint` — ESLint + Prettier check

## Conventions

- All monetary values stored as integers (cents)
- Use `Result<T, E>` pattern for error handling, never throw in service layer
- Database columns: snake_case; TypeScript: camelCase
```

## 配置权限

`Claude Code `采用权限系统，无需请求即可控制其使用的工具。默认模式下，大多数文件写入和所有 `bash` 命令都需要获得批准。对于活跃的开发环境，建议预先批准常用操作。

使用 `/permissions` 打开权限管理器。为 `Claude` 将重复使用的命令添加模式。使用 `Bash(git *)` 允许所有 `git` 命令，使用 `Bash(npm *)` 允许 `npm` 命令，或 `Bash(npx jest *)` 允许特定工具。文件操作可以限定在特定路径。

设置文件控制项目和用户级别的权限。`.claude/settings.json` 已提交到 `Git` 供团队使用。`.claude/settings.local.json` 已被 `Git` 忽略，用于个人权限覆盖。

```json
{
  "permissions": {
    "allow": [
      "Bash(git *)",
      "Bash(npm *)",
      "Bash(npx *)",
      "Read(**/*)",
      "Write(src/**/*)",
      "Edit(src/**/*)"
    ]
  }
}
```

对于生产部署等敏感操作，请将其设置为需要审批，或者在技能上使用 `disable-model-invocation: true` ，这样 `Claude` 就永远无法自动触发它们。

## 安全——市场限制

使用 `blockedMarketplaces` 可以限制可以使用的插件市场。条目支持 `hostPattern` 按域名进行屏蔽（例如， `"*.example.com"` ），以及 `pathPattern` 按仓库路径进行屏蔽（例如， `"acme/corp-plugins"` ）：

```json
{
  "blockedMarketplaces": [
    { "hostPattern": "*.untrusted-domain.io" },
    { "pathPattern": "acme/corp-plugins" }
  ]
}
```

此规则在策略级别强制执行——用户无法通过本地设置覆盖它。适用于企业部署的托管策略。

## 设置和环境

设置遵循官方优先级模型。优先级从高到低依次为：托管设置(`managed settings`)、当前会话的命令行参数、 `.claude/settings.local.json` 、 `.claude/settings.json` 和 `~/.claude/settings.json` 。请注意，本地设置（ `.claude/settings.local.json` ）会覆盖项目设置————个人偏好优先于团队配置，只有托管设置和命令行参数才会覆盖本地设置。托管交付可以使用平台策略文件或托管配置目录，但这些是顶层托管的实现细节，而非独立的日常范围

除了权限设置之外，其他有用的设置还包括： `env` 用于设置每个会话中都应存在的环境变量）、 `agent` 用于设置自定义默认代理）以及 `claudeMdExcludes` 用于过滤单体仓库中无关的内存文件）。您还可以设置默认模型和工作量级别：

```json
{
  "model": "claude-sonnet-4-6",
  "env": {
    "NODE_ENV": "development",
    "LOG_LEVEL": "debug"
  }
}
```

将 `.claude/settings.local.json` 添加到 `.gitignore` 文件中，以确保个人设置保持独立。通过 `Git` 与团队共享 `.claude/settings.json` 、 `CLAUDE.md` 、 `.claude/rules/` 、`.claude/skills/` 以及可选的 `.claude/agents/` 文件。这样，团队成员就能使用相同的项目指令和项目范围的扩展，而个人设置和自动内存管理则保留在每台机器上。


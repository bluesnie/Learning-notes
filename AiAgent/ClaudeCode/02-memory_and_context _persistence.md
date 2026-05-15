###### datetime:2026/05/12 10:41

###### author:nzb

> [参考链接](https://claude.nagdy.me/)

# Memory & Context Persistence

在 `Claude Code` 中，“内存”指的是跨会话持久存在的上下文。与每次启动都会重置的对话窗口不同，内存文件会在每次 `Claude Code` 启动时自动加载。本模块将解释内存文件的层级结构、如何创建和更新内存文件，以及自动内存如何在后台运行。

## 记忆层次结构

`Claude Code` 有两个主要的内存系统：您编写的 `CLAUDE.md` 文件，以及 `Claude` 自行写入的自动内存。官方文档中记录的 `CLAUDE.md` 文件位置包括：
- 管理策略（组织范围）
- 项目说明（ `CLAUDE.md` 或 `.claude/CLAUDE.md` ）
- 用户说明（ `~/.claude/CLAUDE.md` ）和本地说明（ `./CLAUDE.local.md` — 个人项目特定，已添加到 `.gitignore`）。

项目内存(Project memory)是你最常用的。它是一个提交到 `Git` 并与团队共享的 `Markdown` 文件。把你的技术栈、命名规范、常用命令和一些不太明显的陷阱都放在这里。用户内存则用于记录你所有项目中都适用的个人偏好——你偏好的代码模式、你喜欢的代码解释方式以及你常用的工具。

对于大型项目，请将指令拆分到 `.claude/rules/*.md` 文件中。规则可以是全局性的，也可以是针对带有 `frontmatter` 的路径的。例如 `paths: src/api/**/*.ts` 规则仅在 `Claude` 处理匹配的文件时才会生效。

```text
---
paths: src/api/**/*.ts
---
All API endpoints must validate input with Zod. Return 400 with field-level errors on validation failure.
```

## 创建和更新内存

最快的启动方式是使用 `/init` 。在项目目录中运行该命令，`Claude` 将分析代码库并生成一个启动 `CLAUDE.md` 文件。使用 `CLAUDE_CODE_NEW_INIT=1` `claude` 可以实现交互式多阶段设置流程。

对于较大的编辑， `/memory` 会在系统编辑器中打开内存文件。进行更改并保存后，`Claude` 会自动重新加载它们。如果您希望 `Claude` 自动记住某些内容，请直接告诉它，例如“记住 `API` 测试需要 `Redis`”。如果您希望将其写入 `CLAUDE.md` 文件，请明确地告诉 `Claude` 将其添加到该文件中。`@path/to/file` 导入语法允许您引用现有文档，而不是重复编写：

```text
# Project Standards

@README.md
@docs/architecture.md
@package.json
```

导入功能最多支持五层嵌套。首次从外部路径导入时，会触发审批对话框。

## 自动记忆

自动记忆库是一个目录，`Claude` 会在会话期间将自身发现的模式、项目特定行为和调试见解写入其中。 `~/.claude/projects/<project>/memory/MEMORY.md` 文件中的前 200 行或 25KB（以先到者为准）会在会话开始时自动加载。其他主题文件（ `debugging.md` 和 `api-conventions.md` ）则会按需加载。

您无需手动维护自动内存；`Claude` 会自动处理写入操作。如果您想更正或补充 `Claude` 的笔记，可以读取和编辑文件。您可以在 `/memory` 中切换自动内存启用状态，也可以使用 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` `claude` 在当前会话中禁用自动内存，或者在设置中启用 `autoMemoryEnabled` 。要将目录移动到同步位置或自定义路径，请在用户设置中设置 `autoMemoryDirectory` （而不是项目或本地设置——项目和本地设置可能会将写入操作重定向到敏感位置，因此不被接受）：


```json
{
  "autoMemoryEnabled": true,
  "autoMemoryDirectory": "/path/to/shared/memory"
}
```

在包含大量 `CLAUDE.md` 文件的大型单体仓库中，可以使用设置中的 `claudeMdExcludes` 来跳过不相关的文件：

```text
{
  "claudeMdExcludes": ["packages/legacy-app/CLAUDE.md", "vendors/**/CLAUDE.md"]
}
```

`Claude` 还会加载当前工作目录上一级目录下的 `CLAUDE.md` 文件，并在读取子目录中的文件时按需加载这些子目录中的 `CLAUDE.md` 文件。在单体仓库中， `claudeMdExcludes` 有助于将不相关的指令排除在上下文之外



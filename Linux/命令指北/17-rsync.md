###### datetime:2025/10/29 17:30:00

###### author:nzb

# rsync

好的，这是一份非常详细的 `rsync` 命令详解。`rsync` 是一个功能强大、用途广泛的文件同步工具，在 Linux 和 Unix 系统中被广泛使用。

### 一、rsync 是什么？

**rsync** 的全称是 **Remote Sync**，即远程同步。它是一个用于高效地在本地磁盘与远程主机之间，或者两个本地目录之间，同步文件和目录的工具。

它的核心优势在于：
*   **增量同步**：只传输源和目标之间差异的部分，极大地提高了同步效率，尤其是在处理大文件或大量小文件时。
*   **保持属性**：可以完美地保持文件的权限、时间戳、软硬链接、所有者、组信息等。
*   **灵活性**：支持通过 Shell（如 ssh）加密传输，也可以使用自己的守护进程模式（rsync daemon）进行高速同步。
*   **多功能**：可以删除目标端多余的文件，是一个真正的“同步”工具。

---

### 二、核心工作原理：增量传输算法

`rsync` 的核心是其高效的增量传输算法。它并非简单地比较文件日期和大小，而是通过以下步骤实现精确定位和传输差异：

1.  **文件分块**：将源文件分割成一系列固定大小的数据块（通常不是物理分割，而是逻辑计算）。
2.  **计算校验和**：对源文件的每个数据块计算两个校验和：一个强校验和（如 MD5）和一个弱校验和（一种更快的滚动校验）。
3.  **校验和对比**：将源文件的所有校验和发送给目标端。
4.  **目标端查找**：目标端在自己的文件中滑动查找，计算相同大小块的校验和，并与接收到的校验和列表进行比较。
5.  **构建差异集**：目标端告诉源端：“我拥有哪些数据块（通过校验和匹配），请发送我缺失的部分。”
6.  **传输与重组**：源端只传输目标端缺失的数据块和重组文件的指令。目标端利用已有的数据块和新接收的数据块，重新组装出与源端一致的文件。

这种机制使得 `rsync` 在同步大文件（如日志文件、虚拟机磁盘镜像）时效率极高，即使文件中间有少量修改，也只需传输修改部分。

---

### 三、基本语法

```bash
rsync [选项] 源路径 目标路径
```

**路径格式：**
*   **本地路径**：`/path/to/local/dir` 或 `./local_file`
*   **远程路径（通过 SSH）**：
    *   `username@hostname:/path/to/remote/dir` （推送到远程）
    *   `username@hostname:/path/to/remote/file /local/path` （从远程拉取）
*   **远程路径（通过 rsync daemon）**：
    *   `username@hostname::module_name/path/in/module`
    *   `rsync://hostname:port/module/path`

---

### 四、常用选项详解

`rsync` 的选项非常多，以下是日常使用中最核心和最常用的一些。

#### 1. 基础与行为模式
*   `-v, --verbose`：详细模式，输出同步过程中的信息。
*   `-q, --quiet`：安静模式，抑制非错误信息。
*   `-P`：等价于 `--partial --progress`。
    *   `--progress`：显示传输进度。
    *   `--partial`：保留因故中断的临时文件，以便续传。
*   `-r, --recursive`：递归同步目录及其子目录。**（同步目录时必须）**
*   `-a, --archive`：归档模式。这是**最常用的选项**，它等价于 `-rlptgoD`，意味着：
    *   `-r`：递归
    *   `-l`：保持符号链接
    *   `-p`：保持权限
    *   `-t`：保持修改时间
    *   `-g`：保持属组
    *   `-o`：保持属主（需要超级用户权限）
    *   `-D`：保持设备文件和特殊文件
    *   **注意**：`-a` **不保留** 硬链接，需要使用 `-H` 选项。
*   `-z, --compress`：在传输过程中进行压缩，可以节省带宽，但会消耗 CPU。

#### 2. 文件处理与过滤
*   `--delete`：**删除**目标端存在但源端不存在的文件。**（危险但重要！）**
    *   **警告**：使用此选项前务必先用 `--dry-run` 测试，否则可能误删重要文件。
*   `--exclude=PATTERN`：排除符合 PATTERN 的文件或目录。
    *   例如：`--exclude='*.log' --exclude='cache/'`
*   `--include=PATTERN`：指定不排除的模式，通常与 `--exclude` 配合使用。
*   `-n, --dry-run`：模拟运行，只显示会做什么操作，而不会实际执行。**（强烈推荐在首次运行或使用危险选项前使用！）**
*   `-b, --backup`：对目标端即将被覆盖或删除的文件进行备份。
*   `--backup-dir=DIR`：与 `-b` 联用，将备份文件存放到指定目录。
*   `-u, --update`：跳过目标端修改时间比源端新的文件（不覆盖目标端的新内容）。

#### 3. 高级与传输选项
*   `-l, --links`：保持符号链接。
*   `-L, --copy-links`：将符号链接转换为它所指向的实际文件/目录。
*   `-H, --hard-links`：保持硬链接。
*   `-h, --human-readable`：以人类可读的格式输出数字（如 KB, MB）。
*   `-e, --rsh=COMMAND`：指定使用的远程 Shell。
    *   最常用的就是 `-e ssh`，这也是默认行为。你也可以指定 ssh 端口：`-e 'ssh -p 2222'`
*   `--port=PORT`：指定 rsync daemon 的端口。

---

### 五、实用示例

假设本地有一个目录 `/home/user/data`，远程主机是 `remote-server.com`，用户是 `alice`。

#### 1. 本地同步
```bash
# 将 /src 目录同步到 /dest（递归并保持属性）
rsync -av /src/ /dest/

# 注意：源路径尾部的 `/` 很重要！
# rsync -av /src/ /dest/   => 同步 src 目录下的 *所有内容* 到 dest 中
# rsync -av /src  /dest/   => 同步 src 目录 *本身* 到 dest 中，结果为 /dest/src/
```

#### 2. 推送到远程服务器（本地 -> 远程）
```bash
# 基本推送
rsync -avzP /home/user/data/ alice@remote-server.com:/backup/data/

# 使用特定 SSH 端口，并排除 node_modules 目录
rsync -avzP -e "ssh -p 2222" --exclude 'node_modules' /home/user/project/ alice@remote-server.com:/var/www/html/
```

#### 3. 从远程服务器拉取（远程 -> 本地）
```bash
# 基本拉取
rsync -avzP alice@remote-server.com:/backup/data/ /home/user/restored_data/

# 拉取并删除本地多余的文件（让本地成为远程的精确镜像）
rsync -avzP --delete alice@remote-server.com:/var/log/ /home/user/remote_logs_backup/
```

#### 4. 高级用法
```bash
# 1. 模拟运行并删除：看看会删掉哪些文件，确认无误后再执行
rsync -avn --delete /src/ /dest/

# 2. 同步并备份被覆盖的文件到指定目录
rsync -avb --backup-dir=/path/to/backups/$(date +%Y%m%d) /src/ /dest/

# 3. 通过 rsync daemon 同步
rsync -av data/ alice@remote-server.com::backup_module/data/

# 4. 限速同步，避免占用所有带宽（单位：KB/s）
rsync -avz --bwlimit=1000 large_file.iso user@host:/destination/
```

---

### 六、注意事项与最佳实践

1.  **尾随斜杠 `/`**：这是 `rsync` 新手最容易混淆的地方。
    *   `source/`：同步 `source` 目录下的**内容**。
    *   `source`：同步 `source` 目录**本身**。
    *   **建议**：明确你的意图，并在使用前用 `--dry-run` 确认。

2.  **权限问题**：
    *   要保持文件所有者/组（`-o`/`-g`），你通常需要以 `root` 身份运行 `rsync`。
    *   如果远程用户没有写入目标目录的权限，同步会失败。

3.  **`--delete` 的危险性**：这是一个“镜像”操作，会永久删除文件。**永远先和 `-n` 一起使用进行模拟！**

4.  **链接处理**：理解 `-l`（保持链接）和 `-L`（跟随链接）的区别。默认的 `-a` 包含 `-l`。

5.  **测试！测试！测试！**：在执行任何可能造成数据丢失的操作（尤其是涉及 `--delete`）之前，务必使用 `--dry-run` 或 `-n` 选项。

### 总结

`rsync` 是一个极其强大和高效的工具，是系统管理员、开发者和任何需要管理文件的用户的必备利器。掌握其核心概念（增量同步）和常用选项（尤其是 `-a`, `-v`, `-z`, `-P`, `--delete`, `--exclude`, `-n`），你将能够轻松应对各种文件备份、同步和部署任务。
###### datetime:2023/03/30 10:52

###### author:nzb

# container-diff

## 什么是 container-diff

container-diff 是一个用来分析容器镜像的工具，可以分析以下内容

- Docker Image History
- Image file system
- Image size
- Apt packages
- RPM packages
- pip packages
- npm packages

> [container-diff](https://github.com/GoogleContainerTools/container-diff)

## 命令

### 分析单个镜像

- 命令

```text
container-diff analyze <img>     [Run default analyzers]
container-diff analyze <img> --type=history  [History]
container-diff analyze <img> --type=file  [File System]
container-diff analyze <img> --type=size  [Size]
container-diff analyze <img> --type=rpm  [RPM]
container-diff analyze <img> --type=pip  [Pip]
container-diff analyze <img> --type=apt  [Apt]
container-diff analyze <img> --type=node  [Node]
container-diff analyze <img> --type=apt --type=node  [Apt and Node]
# --type=<analyzer1> --type=<analyzer2> --type=<analyzer3>,...
```

- 参数
    - `-j`：JSON 格式化输出
    - `-o --order`：排序，针对 file/package，其余还是按照名称排序
    - `-w`：输出文件
    - `-t --type`：分析类型
- 示例：
  `container-diff analyze registry.cn-hangzhou.aliyuncs.com/quicktron_robot/upper-computer-arm64:GitLabBase_20220621 --type=file -o -w ./result_file_0621.txt`

### 分析多个镜像对比

- 命令

```text
container-diff diff <img1> <img2>     [Run default differs]
container-diff diff <img1> <img2> --type=history  [History]
container-diff diff <img1> <img2> --type=file  [File System]
container-diff diff <img1> <img2> --type=size  [Size]
container-diff diff <img1> <img2> --type=rpm  [RPM]
container-diff diff <img1> <img2> --type=pip  [Pip]
container-diff diff <img1> <img2> --type=apt  [Apt]
container-diff diff <img1> <img2> --type=node  [Node]
```

- 参数同上
- 示例
  `container-diff diff registry.cn-hangzhou.aliyuncs.com/quicktron_robot/upper-computer-arm64:GitLabBase_20230328 registry.cn-hangzhou.aliyuncs.com/quicktron_robot/upper-computer-arm64:GitLabBase_20220621 --type=file -o -w ./result_file_20230328_20220621.txt`

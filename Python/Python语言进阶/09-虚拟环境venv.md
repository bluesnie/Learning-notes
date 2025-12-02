###### datetime:2024/11/08 15:52

###### author:nzb

# Python venv模块

## 使用虚拟环境

venv模块用于为Python项目创建与管理互相分离的虚拟环境。

这有助于:
- 帮助其他程序员重建你的开发设置
- 避免项目依赖冲突。

venv模块是 Python 标准库的一部分，自 Python 3.5 以来一直是官方推荐的创建虚拟环境的方法。

> 注：还有其他创建虚拟环境的优秀第三方工具，如 conda 和 virtualenv。这些工具中的任何一个都可以帮助您建立虚拟环境，而且不止于此。

本文均以Linux系统为例进行说明。

### 创建

当处理一个使用 pip 安装的外部依赖项的 Python 项目，最好先创建一个虚拟环境。

```shell
python3 -m venv venv/
```

命令中使用的第一个venv指定了模块（即运行venv模块），第二个venv/则设置了虚拟环境的名称，可以给它起不同的名字，但惯例是叫它venv。

> 注：虚拟环境名称的末尾加上斜线 (/)是不必要的，但它可以提醒你正在创建一个文件夹。

### 激活(activate)

在创建完虚拟环境后需激活该环境，在运行创建虚拟环境命令的同路径下运行：

```shell
source venv/bin/activate
```

> 如果虚拟环境命名并非venv，则需使用虚拟环境名替换venv
>
> 也可不激活环境直接使用，这样的话需使用虚拟环境Python解释器的绝对路径来执行Python程序。

激活后命令行前面会显示虚拟环境名，如(venv)。

### 在虚拟环境中安装Python包

```shell
(venv) $ python -m pip install <package-name>
```

当创建并激活虚拟环境后，pip将把包安装在隔离的位置。

> 由于创建虚拟环境时使用的是 Python 3 版本，因此无需明确调用 python3 或 pip3。只要虚拟环境处于激活状态，python 和 pip 就会链接到与 python3 和 pip3 相同的可执行文件。

只要不关闭终端，安装的每个 Python 软件包最终都会进入这个隔离环境，而不是的全局 Python 软件包。这意味着你现在可以在 Python 项目中工作，而不必担心依赖关系冲突。

### 停用环境(deactivate)

```shell
(venv) $ deactivate
$
```

该命令将退出虚拟环境，使环境恢复到全局环境。可重新激活。


## 为何要使用虚拟环境

直接原因：Python本身依赖管理不佳——若未具体指定，pip将把所有你安装的外部包都置于名为site-package的文件夹下。

>从技术上讲，Python 有两个 site-packages 文件夹：
>
>- purelib/ 只包含用纯 Python 代码编写的模块。
>
>- platlib/ 应包含非纯 Python 编写的二进制文件，例如 .dll、.so 或 .pydist 文件。

如果你使用的是 Fedora 或 RedHat Linux 发行版，你可以在不同的位置找到这些文件夹。

不过，大多数操作系统都会执行 Python 的 site-packages 设置，使这两个位置指向相同的路径，从而有效地创建一个单一的 site-packages 文件夹。你可以使用 sysconfig 检查路径：

```shell
>>> import sysconfig
>>> sysconfig.get_path("purelib")
'/usr/local/lib/python3.12/site-packages'
>>> sysconfig.get_path("platlib")
'/usr/local/lib/python3.12/site-packages'
```

具体而言，虚拟环境解决的问题包括：

### 避免系统污染

Linux 和 macOS 预装了操作系统用于内部任务的 Python 版本。

如果将软件包安装到操作系统的全局 Python 中，这些软件包将与系统相关的软件包混合。这种混合可能会对操作系统正常运行的关键任务产生意想不到的副作用。

此外，如果你更新了操作系统，那么你安装的软件包可能会被覆盖或丢失。

### 避免依赖冲突

一个项目可能需要与另一个项目不同版本的外部库。如果你只有一个安装包的地方，那么你就无法使用同一个库的两个不同版本。

解决方案则说为存在依赖冲突的不同项目分别创建一个虚拟环境。

### 尽量减少可重复性问题

将所有包都安装在同一位置，则很难区分每个单独项目的依赖，进行项目的复制或迁移的时候会十分麻烦。

若为每个项目都使用一个单独的虚拟环境，那么就可以更直观地从固定的依赖关系中读取项目需求。这意味着，当你开发出一款出色的应用程序时，你可以更轻松地分享你的成功经验，让其他人有可能与你合作！

### 避免安装权限锁定

最后，你可能需要计算机上的管理员权限，才能将软件包安装到 Python 主机的 site-packages 目录中。在公司工作环境中，你很可能缺乏相应权限。

如果你使用虚拟环境，那么你就可以在用户权限范围内创建一个新的安装位置，这样你就可以安装和使用外部软件包。

无论你是作为业余爱好在自己的机器上编码，还是为客户开发网站，抑或是在公司环境中工作，使用虚拟环境从长远来看都会为你省去很多麻烦。

## 虚拟环境是什么

简单而言，Python虚拟环境是一个文件夹结构，它提供轻量且隔离的Python环境所需的一切。

### 一个文件夹结构

当使用venv模块创建虚拟环境时，Python创建一个自包含的文件夹，并复制或连接Python可执行文件到该文件夹结构下。

一个虚拟环境文件夹包含非常多的文件和文件夹，但是大部分都位于site-packages/文件夹下，如果将对应文件夹下的子文件夹和文件都忽略，则整体结构不算太冗长：

```bash
venv/
│
├── bin/
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.csh
│   ├── activate.fish
│   ├── pip
│   ├── pip3
│   ├── pip3.12
│   ├── python
│   ├── python3
│   └── python3.12
│
├── include/
│   │
│   └── python3.12/
│
├── lib/
│   │
│   └── python3.12/
│       │
│       └── site-packages/
│           │
│           ├── pip/
│           │
│           └── pip-24.2.dist-info/
│
├── lib64/
│   │
│   └── python3.12/
│       │
│       └── site-packages/
│           │
│           ├── pip/
│           │
│           └── pip-24.2.dist-info/
│
└── pyvenv.cfg
```

- `bin/`包含虚拟环境的可执行文件。最值得注意的是 Python 解释器 (python) 和 pip 可执行文件 (pip)，以及它们各自的符号链接（python3、python3.12、pip3、pip3.12）。该文件夹还包含虚拟环境的激活脚本。具体的激活脚本取决于使用的 shell。
- `include/`是一个初始为空的文件夹，Python 用它来包含 C 头文件，以便安装依赖于 C 扩展的软件包。
- `lib/`包含嵌套在指定 Python 版本 (python3.12/) 文件夹中的 site-packages/ 目录。虚拟环境中使用的外部软件包将安装在该文件夹中。从 Python 3.12 开始，虚拟环境只预装了一个依赖包 pip。
- `lib64：`出于兼容性考虑，许多 Linux 系统中的lib64/都是以lib/的符号链接形式出现的。某些 Linux 系统可能会根据其体系结构使用`lib/`和`lib64/`之间的区别来安装不同版本的库。
- `{name}-{version}.dist-info/`是pip的默认目录，包含软件包分发信息，用于记录已安装软件包的相关信息。
- `pyvenv.cfg`是虚拟环境的关键文件。它只包含几个键值对，Python 使用它们来设置 sys 模块中的变量，这些变量决定了当前 Python 会话将使用哪个 Python 解释器和站点软件包目录。

总之，宏观而言，一个虚拟环境有三个主要部分：

- Python二进制文件的副本或符号链接
- `pyvenv.cfg`文件
- `site-packages`目录

`site-packages/`内的安装包是可选的，但基本是默认存在的。即使该目录为空，虚拟环境仍然有效。

在默认设置下，venv 将只安装 pip，这是安装 Python 软件包的推荐工具，毕竟安装其他软件包是虚拟环境最常见的使用情况。

> 若Python版本<3.12，则site-packages/目录下会有一些额外的文件夹：
> 
> - `setuptools` 模块： Python 打包生态系统中的基本工具。setuptools 扩展了 distutils 模块，提供了软件包发现、分发和依赖管理等功能。将 setuptools 作为默认设置可确保用户无需安装其他工具就能立即使用这些功能。在采用 PEP 517 和 PEP 518 之前，许多软件包都依赖 setuptools 进行安装。
> 
> - `_distutils_hack/` 模块确保 Python 在安装软件包时，选择本地的 ._distutils子模块而不是标准库的 distutils 模块。
> 
> - `pkg_resources/` 模块帮助应用程序自动发现插件，并允许 Python 软件包访问它们的资源文件。它与 setuptools 一起发布。
> 
> 最后，还有一个名为 `distutils-precedence.pth` 的文件。该文件帮助设置 `distutils` 导入的路径优先级，并与 `_distutils_hack` 一起确保 `Python` 优先使用与 `setuptools` 捆绑的 `distutils` 版本，而不是内置版本。
> 
> 若 Python 版本>=3.12，那么这些文件与文件夹不会预装在虚拟环境中。即使使用的是旧版本的 Python，也不需要记住它们就能有效地使用虚拟环境。
> 
> 只需记住，在 `site-packages/` 目录中预装的软件包都是标准工具，它们能让安装其他软件包变得更方便。

综上，Python虚拟环境只是一个文件架构，可以随时删除和重建。

隔离的Python安装
为构建一个隔离的环境以免安装的外部包与全局包冲突，venv重建了标准Python安装会创建的文件夹结构。

> 如一个文件夹结构一节中指出的，该结构包括Python 二进制文件的副本或 symlink 的位置和 site-packages 目录（Python 安装外部软件包的目录）。
> 
> 虚拟环境中的 Python 可执行文件是环境所基于的 Python 可执行文件的副本还是符号链接，主要取决于操作系统。
> 
> Windows 和 Linux 可能会创建符号链接而不是副本，而 macOS 则总是创建副本。可尝试在创建虚拟环境时使用可选参数来影响默认行为。在大多数情况下，这个问题没啥影响。

值得注意的是，Python标准库模块并未包含于虚拟环境文件夹结构下，但依然可以在虚拟环境调用。这是因为虚拟环境重用了用于创建虚拟环境的 Python 安装中的 Python 内置模块和标准库模块。

> 因为创建虚拟环境总是需要一个现有的 Python 安装，venv 选择重用现有的标准库模块，以避免将它们复制到新虚拟环境的开销。正如 PEP 405 的动机所述，这种有意为之的行为加快了虚拟环境的创建速度，并使其更加轻量级。

除了标准库模块外，你还可以在创建环境时通过参数让虚拟环境访问基本安装的软件包：

```shell
$ python3 -m venv venv/ --system-site-packages
```

如果在调用 venv 时添加 --system-site-packages，Python 将把 pyvenv.cfg 中 include-system-site-packages 的值设置为 true。这一设置意味着你可以使用安装到基本 Python 中的任何外部软件包，就像将它们安装到虚拟环境中一样。

这种连接是单向的。即使你允许虚拟环境访问源 Python 的 site-packages 文件夹，你安装到虚拟环境中的任何新包也不会与那里的包混合。Python 会尊重安装到虚拟环境中的软件包的隔离性，并将它们放到虚拟环境中独立的 site-packages 目录中。

总而言之，虚拟环境基本上就是一个带有设置文件的文件夹结构。它可能预装了 pip，也可能没有，它可以访问源 Python 标准库模块，同时保持隔离。

## 虚拟环境如何工作

### 复制结构与文件

当使用venv创建虚拟环境时，模块会重建操作系统上标准Python安装的文件和文件夹结构，从而保证Python可以如预期一样隔离地工作，而无需额外的改变。Python还会将调用venv模块时使用到的Python可执行文件的副本或符号链接也复制到该文件夹结构中：

```shell
venv/
│
├── bin/
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.csh
│   ├── activate.fish
│   ├── pip
│   ├── pip3
│   ├── pip3.12
│   ├── python
│   ├── python3
│   └── python3.12
│
├── include/
│
├── lib/
│   │
│   └── python3.12/
│       │
│       └── site-packages/
│
├── lib64/
│   │
│   └── python3.12/
│       │
│       └── site-packages/
│
└── pyvenv.cfg
```

> 如果你找到系统安装Python的位置，并观察其目录结构，你会发现虚拟环境与其十分相似。
>
> 虚拟环境基于的基础Python安装在 `pyvenv.cfg` 文件中的home键下可见。

### 调整前缀查找(Prefix-Finding)过程

有了标准的文件夹结构，只需根据 venv 规范对其前缀查找过程稍作调整，虚拟环境中的 Python 解释器就能了解所有相关文件的位置。

Python 解释器不是通过查找 os 模块来确定标准库的位置，而是首先查找 `pyvenv.cfg` 文件。如果解释器找到了这个文件，并且其中包含一个 home 关键字，那么解释器就会使用这个关键字来设置两个变量的值：

- `sys.base_prefix` 将保存用于创建此虚拟环境的 Python 可执行文件的路径，您可以在 `pyvenv.cfg` 中的 home 关键字下定义的路径找到它。
- `sys.prefix` 将指向包含 `pyvenv.cfg` 的目录。

如果解释器找不到 `pyvenv.cfg` 文件，那么它就会认为自己不是在虚拟环境中运行，这时 `sys.base_prefix` 和 `sys.prefix` 都会指向相同的路径。

> 可通过查看sys.base_prefix和sys.prefix变量内容来验证该过程。
> ```python
> >>> import sys
> >>> sys.base_prefix
> >>> sys.prefix
> ```
> 激活环境后：
>
> ```python
> >>> import sys
> >>> sys.base_prefix
> '/usr/local'
> >>> sys.prefix
> '/home/name/path/to/venv'
> ```
> 停用环境后：
> ```python
> >>> import sys
> >>> sys.base_prefix
> '/usr/local'
> >>> sys.prefix
> '/usr/local'
> ```

若上述两个变量值不同，则Python调整寻找模块的路径：

`site`和`sysconfig`标准库模块被修改以使标准库和头文件将相对于 sys.base_prefix […] 查找，而 site-package 目录 […] 仍相对于 sys.prefix […] 查找。

这一更改允许虚拟环境中的 Python 解释器使用基本 Python 安装中的标准库模块，同时指向其内部 site-packages 目录来安装和访问外部软件包。

### 链接回标准库

Python虚拟环境旨在提供一个轻量的提供隔离的Python环境的方法，从而可以快速地创建或删除。由此，`venv` 只复制最小化的必要文件。

虚拟环境中的 Python 可执行文件可以访问作为环境基础的 Python 安装的标准库模块。Python 通过在 `pyvenv.cfg` 中的 `home` 设置中指向基本 Python 可执行文件的文件路径来实现这一点：

```cfg
home = /usr/local/bin
include-system-site-packages = false
...
```

Python 通过将相关路径添加到 sys.path 来查找标准库模块。在初始化过程中，Python 会自动导入 site 模块，并为该参数设置默认值。

Python 会话在 sys.path 中可以访问的路径决定了 Python 可以从哪些位置导入模块。

如果激活虚拟环境并输入 Python 解释器，则可以确认基本 Python 安装的标准库文件夹路径可用：

```python
>>> import sys
>>> sys.path
['',
 '/usr/local/lib/python312.zip',
 '/usr/local/lib/python3.12',
 '/usr/local/lib/python3.12/lib-dynload',
 '/home/name/path/to/venv/lib/python3.12/site-packages']
```

因为包含标准库模块的目录路径在 sys.path 中可用，所以在虚拟环境中使用 Python 时，可以导入任何标准库模块。

### 修改PYTHONPATH

为确保使用虚拟环境中的 Python 解释器来运行的脚本，venv 会修改 PYTHONPATH 环境变量（可以使用 sys.path 访问该变量）。

若未激活任何虚拟环境，访问该变量，则会看到默认Python安装的路径：

```python
>>> import sys
>>> sys.path
['',
 '/usr/local/lib/python312.zip',
 '/usr/local/lib/python3.12',
 '/usr/local/lib/python3.12/lib-dynload',
 '/usr/local/lib/python3.12/site-packages']
```

其中`'/usr/local/lib/python3.12/site-packages'`为`site-packages`目录，该目录包含了安装的外部包（如，使用pip安装的包）。在没有激活虚拟环境的情况下，该目录嵌套在与 Python 可执行文件相同的文件夹结构中。

> Windows 上的Roaming文件夹包含一个额外的 site-packages 目录，该目录与使用 pip 的 --user 标志的安装相关。该文件夹提供了一定程度的虚拟化，但仍将所有 --user 安装的软件包集中在一处。

若激活了虚拟环境，则sys.path变量值改变：

```python
>>> import sys
>>> sys.path
['',
 '/usr/local/lib/python312.zip',
 '/usr/local/lib/python3.12',
 '/usr/local/lib/python3.12/lib-dynload',
 '/home/name/path/to/venv/lib/python3.12/site-packages']
```

即Python使用虚拟环境下的`site-package`目录路径更换了默认的路径，使得Python加载虚拟环境中的外部包。而且由于全局的`site-packages`目录未被列出，故Python不会加载其中的模块。

> 在 Windows 系统中，Python 还会将虚拟环境的根文件夹路径添加到 sys.path。

由此，Python实现了虚拟环境中的外部包隔离。

> 此外，可以在创建虚拟环境时传递一个参数，从而实现以只读方式访问基本 Python 安装的系统 site-packages 目录。

### 激活时改变Shell的PATH变量

虽然不是必须的，但出于一致性考虑，一般推荐在在虚拟环境中工作之前先激活虚拟环境：

```bash
$ source venv/bin/activate
(venv) $
```

> 具体使用的激活脚本取决于操作系统以及使用的shell。
> 
> 如果深挖虚拟环境的目录结构，将发现其附带了一些不同的激活脚本：
> ```shell
> venv/
> │
> ├── bin/
> │   ├── Activate.ps1
> │   ├── activate
> │   ├── activate.csh
> │   ├── activate.fish
> │   ├── pip
> │   ├── pip3
> │   ├── pip3.12
> │   ├── python
> │   ├── python3
> │   └── python3.12
> │
> ├── include/
> │
> ├── lib/
> │
> ├── lib64/
> │
> └── pyvenv.cfg
> ```
> 这些激活脚本的作用是一样的。不同脚本用于适应不用的操作系统以及shell。

激活脚本有两个主要行为：

- 路径：将`VIRTUAL_ENV`变量的值设定为虚拟环境的根目录，并将其 Python 可执行文件的相对位置预置到 PATH 中；
- 命令提示符：由于脚本更改了命令提示符（在前面添加环境名），因此您可以很快知道虚拟环境是否已激活。

这两项更改都是并非必要的小改动，纯粹是为了方便使用。

当停用虚拟环境后，shell将会把改变恢复回去。

### 可在任意位置通过绝对路径使用

如前面的章节所述，无需激活虚拟环境即可使用之。

> 当在shell中只提供了二进制文件的名字时，shell将在PATH记录的路径中搜寻拥有该名字的可执行文件，选出并运行第一个批判的文件。

激活脚本更改了PATH变量以使shell首先在虚拟环境的二进制文件目录下搜寻可执行文件，从而使得用户可以仅用pip或python这个名字便运行虚拟环境中的相应文件。

如果不激活环境的话，则可以通过使用对应Python的绝对路径来运行虚拟环境中的任何脚本：

```shell
$ /home/name/path/to/venv/bin/python
```
这个与激活环境再用python 调用是等价的。

## venv进阶操作

修改命令提示符（环境名）

如前面章节所言，虚拟环境名是可以任取的。常用的有：
- venv
- env
- .venv

> 使用venv作为虚拟环境名是一个惯例，它有助于使用.gitignore文件从版本控制中可靠地排除虚拟环境

使用自定义的环境名：

```python
python3 -m venv your-fancy-name/
source your-fancy-name/bin/activate
```

如前面的章节所言，这会在当前路径下创建`your-fancy-name`文件夹，其中包含了类似标准Python安装的结构，而且由于运行了激活脚本，命令行提示符最前面会显示自定义的虚拟环境名。

若想要使命令提示符前面显示的虚拟环境名与实际创建的虚拟环境文件夹名不同，则可用`--prompt`参数指定：

```shell
$ python3 -m venv venv/ --prompt dev-env
$ source venv/bin/activate
(dev-env) $
```

### 覆盖已有环境

若创建一个环境后再次在相同位置创建同名环境，则新环境不会覆盖原来的环境。

例如：

```shell
$ python3 -m venv venv/
$ venv/bin/pip install requests
$ venv/bin/pip list
Package            Version
------------------ ---------
certifi            2024.8.30
charset-normalizer 3.3.2
idna               3.8
pip                24.2
requests           2.32.3
urllib3            2.2.2

$ python3 -m venv venv/
$ venv/bin/pip list
Package            Version
------------------ ---------
certifi            2024.8.30
charset-normalizer 3.3.2
idna               3.8
pip                24.2
requests           2.32.3
urllib3            2.2.2
```

在本例中，第一次创建的环境中安装的requests包并未因为再次创建相同环境而被覆盖。(本例中，使用虚拟环境中pip的绝对路径，而未激活虚拟环境，这与激活环境后再使用pip是等效的。)

要覆盖原同名环境，则需添加`--clear`参数：

```shell
$ python3 -m venv venv/
$ venv/bin/pip install requests
$ venv/bin/pip list
Package            Version
------------------ ---------
certifi            2024.8.30
charset-normalizer 3.3.2
idna               3.8
pip                24.2
requests           2.32.3
urllib3            2.2.2

$ python3 -m venv venv/ --clear
$ venv/bin/pip list
Package    Version
---------- -------
pip        24.2
```

### 一次创建多个虚拟环境

调用venv模块时后面可跟多个路径，即可同时创建多个虚拟环境(后面跟的虚拟环境数没有直接限制)：

```shell
$ python3 -m venv venv/ /home/name/virtualenvs/venv-copy/
```

> 实际上，`python3 -m venv`后面跟的路径是shell里面的常规路径，即可以是相对路径和绝对路径，可创建在任意有对应权限的位置。venv/也算是一种相对路径，与./venv/等价。

### 更新核心依赖

当新创建一个虚拟环境并使用pip安装外部包时，可能会遇到一个警告信息：

```shell
WARNING: You are using pip version 23.2.4; however, version 24.2 is available.
You should consider upgrading via the
'/path/to/venv/python -m pip install --upgrade pip' command.
```

新建的环境的pip包居然已经过时了。这是因为使用venv创建虚拟环境时安装pip的默认配置时使用`ensurepip`来把pip引入虚拟环境。

但`ensurepip`并不联网，而是使用与每个新发布的`CPython`绑定的pip轮子，由此，绑定的pip与独立的pip项目的更新周期并不同。

而当使用pip安装外部包时，程序会链接`PyPI`并验证pip本身是否过时，若过时，则显示上述警告。

可在遇到该警告时按提示更新，也可在创建虚拟环境时直接加上参数`--upgrade-deps`：

```shell
$ python3 -m venv venv/ --upgrade-deps
$ source venv/bin/activate
(venv) $ python -m pip install --upgrade pip
Requirement already satisfied: pip in ./venv/lib/python3.12/site-packages (24.2)
```

使用该参数会在创建环境时自动连接PyPI来更新最新pip版本。

### 不安装pip

创建虚拟环境的主要时间和空间开销都在于pip的安装。一般而言，我们创建虚拟环境后都会用到pip 来安装外部包。若出于一些原因，你不需要在虚拟环境中使用pip，则可以使用`--without-pip`参数：

```shell
$ python3 -m venv venv/ --without-pip
$ du -hs venv/
56K venv
```

这样虚拟环境还是能够提供带有独立Python可执行程序的轻量隔离环境。

> 不在虚拟环境中安装pip，可能仍能使用pip命令，但这并不能在虚拟环境中安装外部包，而会把包安装在其他位置，而且无法在虚拟环境中使用。

不在虚拟环境中安装pip，而又仍需安装外部包，可以手动把包安装在虚拟环境的site-packages目录下，或把zip文件放在哪里，并通过`Python ZIP imports`引用。

### 引入系统包

如果你在系统中费时费力费空间地在全局Python环境中安装了一些外部包（如PyTorch或TensorFlow），你可能希望在虚拟环境中使用系统安装的外部包，而不是在虚拟环境中再装一遍。

可在创建虚拟环境时使用`--system-site-packages`标志来实现这一点（它会把虚拟环境的`pyvenv.cfg`文件中的`include-system-site-packages`的值设为true）。这会使你在虚拟环境中具有对系统安装包的只读权限，虚拟环境中安装的包依然在虚拟环境的文件夹下。

### 明确选择创建Python可执行文件的副本或符号链接

创建虚拟环境时，安装的Python可执行文件可能是副本也可能是副本：

- Windows 可以创建符号链接或副本，但某些版本不支持符号链接。创建符号链接可能需要管理员权限。
- Linux 发行版可以创建符号链接或副本，但通常选择符号链接而非副本。
- macOS 总是创建二进制文件的副本。

创建符号链接有助于和系统Python版本保持同步。如更新了系统Python版本，符号链接则同步指向新版本，而副本则仍为原版本。但在Windows下使用符号链接，双击链接时系统可能会优先解析符号链接而忽视虚拟环境。

可使用`--symlinks`或`--copies`标志明确选择时选用符号链接还是副本：

- `--symlinks`将尝试创建符号链接，而不是拷贝。该选项对 macOS 的构建没有任何影响。
- `--copies`将尝试创建 Python 二进制文件的拷贝，而不是将它们链接到基本 Python 安装的可执行文件。

### 将虚拟环境中的Python更新至系统版本

若创建虚拟环境时选用的是Python的副本，而且之后更新了系统的Python版本，则可使用--upgrade标志将虚拟环境中的Python可执行文件和pip 更新至系统相同版本，同时保持其他安装的包不变：

```bash
$ python3 -m venv venv/ --upgrade
```

## Virtualenv environment和conda environment有什么不同

virtualenv 和 conda 都是用于创建和管理 Python 虚拟环境的工具，它们有一些不同之处。下面是它们的主要区别：

### Virtualenv:
- 独立性： virtualenv 创建的虚拟环境相对较轻量，它主要通过在项目目录中创建一个包含 Python 解释器和相关库的目录来实现。这使得虚拟环境相对独立，不会影响系统的其他部分。
- 包管理： virtualenv 主要依赖于 pip 来安装和管理 Python 包。

### Conda:
- 跨平台性： conda 是一个跨平台的环境管理工具，可以用于管理不仅仅是 Python 包，还包括其他语言的包。它可以在 Windows、Linux 和 macOS 上运行。
- 环境复制： Conda 创建的环境包含了整个环境的快照，而不仅仅是 Python 环境。这使得在不同系统之间更容易复制整个环境。
- 包管理： conda 使用自己的包管理系统，因此在使用 conda 环境时，你可以使用conda install 而不仅仅是 pip。

> - 直接使用`conda`的虚拟环境`python`,则不用事先激活环境，会自动激活`/home/blues/anaconda3/envs/speech_interaction/bin/python auto_startup.py`
> - 创建虚拟环境后，`pip list`包含很多包，原因可能是用户目录下存在`/home/blues/.local/lib/python3.10/site-packages`，删除后重新创建。

### 选择虚拟环境类型：
- 项目需求： 如果你的项目主要依赖于 Python 包，而且你想保持环境相对轻量，virtualenv 是一个不错的选择。
- 多语言需求或复杂环境： 如果你的项目需要其他语言的库，或者你需要创建包含多个软件包的复杂环境，conda 可能更适合。

### 指定虚拟环境：
当你在 PyCharm 中为项目指定虚拟环境时，你需要选择符合你项目需求的虚拟环境管理工具。在 “Python Interpreter” 设置中，你可以选择 “Virtualenv Environment” 或 “Conda Environment”，具体取决于你的项目配置。
- 如果你选择 “Virtualenv Environment”，PyCharm 将使用 virtualenv 创建和管理虚拟环境。
- 如果你选择 “Conda Environment”，PyCharm 将使用 conda 创建和管理虚拟环境。

选择哪个取决于你的偏好和项目的需求。如果你的项目已经使用了其中一个，最好保持一致性。

> - conda 可以管理多个环境，每个环境可以包含不同的 **Python 版本**和包。
> - ros 环境中，conda里面的python版本可能跟ros 的版本不一致，会导致解释器的动态库不匹配，导致程序无法运行，这时可以使用 `virtualenv`。
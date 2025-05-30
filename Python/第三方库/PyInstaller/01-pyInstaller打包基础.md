###### datetime:2022/04/16 17:10

###### author:nzb

# pyinstaller

> [官方文档](https://pyinstaller.readthedocs.io/en/v3.3.1/usage.html)

## 常用参数

- `-F, –onefile`：打包一个单个文件，如果你的代码都写在一个.py文件的话，可以用这个，如果是多个.py文件就别用
- `-D, –onedir`：打包多个文件，在dist中生成很多依赖文件，适合以框架形式编写工具代码，我个人比较推荐这样，代码易于维护
- `-K, –tk`：在部署时包含 TCL/TK
- `-a, –ascii`：不包含编码.在支持Unicode的python版本上默认包含所有的编码.
- `-d, –debug`：产生debug版本的可执行文件
- `-w,–windowed,–noconsole`：使用Windows子系统执行.当程序启动的时候不会打开命令行(只对Windows有效)
- `-c,–nowindowed,–console`：使用控制台子系统执行(默认)(只对Windows有效)
    - pyinstaller -c xxxx.py - pyinstaller xxxx.py --console
- `-s,–strip`： 可执行文件和共享库将run through strip.注意Cygwin的strip往往使普通的win32 Dll无法使用.
- `-X, –upx`：如果有UPX安装(执行Configure.py时检测),会压缩执行文件(Windows系统中的DLL也会)(参见note)
- `-o DIR, –out=DIR`：指定spec文件的生成目录,如果没有指定,而且当前目录是PyInstaller的根目录,会自动创建一个用于输出(spec和生成的可执行文件)
  的目录.如果没有指定,而当前目录不是PyInstaller的根目录,则会输出到当前的目录下.
- `-p DIR, –path=DIR`：设置导入路径(和使用PYTHONPATH效果相似).可以用路径分割符(Windows使用分号,Linux使用冒号)
  分割,指定多个目录.也可以使用多个-p参数来设置多个导入路径，让pyinstaller自己去找程序需要的资源
- `–icon=<FILE.ICO>`：将file.ico添加为可执行文件的资源(只对Windows系统有效)，改变程序的图标 pyinstaller -i ico路径 xxxxx.py
- `–icon=<FILE.EXE,N>`：将file.exe的第n个图标添加为可执行文件的资源(只对Windows系统有效)
- `-v FILE, –version=FILE`：将verfile作为可执行文件的版本资源(只对Windows系统有效)
- `-n NAME, –name=NAME`：可选的项目(产生的spec的)名字.如果省略,第一个脚本的主文件名将作为spec的名字

## 通用参数

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `-h, --help` |  显示帮助 | 无 |
| `-v, --version` |  显示版本号 | 无 |
| `–-distpath DIR` |  生成文件放在哪里 | 默认：当前目录的dist文件夹内 |
| `--workpath WORKPATH` |  生成过程中的中间文件放在哪里 | 默认：当前目录的build文件夹内 |
| `-y, --noconfirm` |  如果dist文件夹内已经存在生成文件，则不询问用户，直接覆盖 | 默认：询问是否覆盖 |
| `--upx-dir UPX_DIR` |  UPX_DIR 指定upx工具的目录 默认：execution | path |
| `-a, --ascii` |  不包含unicode支持 | 默认：尽可能支持unicode |
| `–-clean` |  在本次编译开始时，清空上一次编译生成的各种文件 | 默认：不清除 |
| `--log-level LEVEL ` |  控制编译时pyi打印的信息 一共有5个等级，由低到高分别为TRACE DEBUG INFO(默认) WARN ERROR CRITICAL。 | 默认INFO，不打印TRACE和DEBUG信息 |

## 与生成结果有关的参数

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `-D, --onedir` |  生成one-folder的程序（默认） | 生成结果是一个目录，各种第三方依赖、资源和exe同时存储在该目录 |
| `-F, --onefile` |  生成one-file的程序 | 生成结果是一个exe文件，所有的第三方依赖、资源和代码均被打包进该exe内 |
| `--specpath DIR` |  指定.spec文件的存储路径 | 默认：当前目录 |
| `-n NAME, --name NAME` |  生成的.exe文件和.spec的文件名 | 默认：用户脚本的名称，即main.py和main.spec |

## 指定打包哪些资源、代码

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `--add-data <SRC;DEST or SRC:DEST>` | 打包额外资源 用法：`pyinstaller main.py –add-data=src;dest` | windows以;分割，linux以:分割，可多次使用 |
| `--add-binary <SRC;DEST or SRC:DEST>` |  打包额外的代码 | 用法：同–add-data。与–add-data不同的是，用binary添加的文件，pyi会分析它引用的文件并把它们一同添加进来 |
| `-p DIR, --paths DIR` |  指定额外的import路径，类似于使用PYTHONPATH | 参见PYTHONPATH |
| `--hidden-import MODULENAME, --hiddenimport MODULENAME` | 打包额外py库pyi在分析过程中，有些`import`没有正确分析出来，代码中使用`__import__`或`importlib`导包，运行时会报`import error`，这时可以使用该参数 |  |
| `--additional-hooks-dir HOOKSPATH` |  指定用户的hook目录 | hook用法参见其他，系统hook在PyInstaller\hooks目录下 |
| `--runtime-hook RUNTIME_HOOKS` |  指定用户runtime-hook | 如果设置了此参数，则runtime-hook会在运行main.py之前被运行 |
| `--exclude-module EXCLUDES` |  需要排除的module | pyi会分析出很多相互关联的库，但是某些库对用户来说是没用的，可以用这个参数排除这些库，有助于减少生成文件的大小 |
| `--key KEY` |  pyi会存储字节码，指定加密字节码的key | 16位的字符串 |

- 为什么要使用 --add-data？程序里文件格式有很多种：
    - 源代码 .py
    - 图片格式 .png .jpg .ico 等
    - 配置文件 .ini .json .xml等
    - 其他可执行文件 .bin .exe等
    - 模型文件 .pth 等
    - 说明文档 .txt .md等
- 注意：
    - 除了.py之外，其他格式不会编译。
    - 除了.py之外，其他格式若要打包进去，需要使用 --add-data 处理，或者手动拷贝(嫌麻烦，你每次都能记住？)

- 如何使用 --add-data?
    - 用法：pyinstaller x.py --add-data="源地址;目标地址"。 windows以;分割，linux以:分割
    - 例如：将 config 目录的所有文件打包到目标的 config 文件夹（不存在会自动创建）下

```text
 pyinstaller x.py --add-data ".\\config\\*;.\\config"
```

- 可使用多次 --add-data

```text
pyinstaller x.py  -n Demo2.0.3 --key !@)v -i "res\logo.ico"  
--add-data=".\*.txt;." --add-data=".\*.json;." --add-data="res\*.*;.\res" 
--add-data="dist\models\*.*;.\models"
```

## 生成参数

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `-d, --debug` |  执行生成的main.exe时，会输出pyi的一些log，有助于查错 | 默认：不输出pyi的log |
| `-s, --strip` |  优化符号表 | 原文明确表示不建议在windows上使用 |
| `--noupx` |  强制不使用upx | 默认：尽可能使用。 |

## 其他

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `--runtime-tmpdir PATH` |  指定运行时的临时目录 | 默认：使用系统临时目录 |

## Windows和Mac特有的参数

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `-c, --console, --nowindowed` |  显示命令行窗口，与-w相反 | 默认含有此参数 |
| `-w, --windowed, --noconsole` |  不显示命令行窗口 | 编写GUI程序时使用此参数有用。 |
| `-i <FILE.ico or FILE.exe,ID or FILE.icns>, --icon <FILE.ico or FILE.exe,ID or FILE.icns>` |  为main.exe指定图标 `pyinstaller -i beauty.ico` | main.py |

## Windows特有的参数

| 参数名 |  描述 | 说明 |
| ----- |  ------ | ------ |
| `--version-file FILE` | 添加版本信息文件 `pyinstaller –version-file ver.txt` |  |
| `-m <FILE or XML>, --manifest <FILE or XML>` |  添加manifest文件 `pyinstaller -m main.manifest` |  |
| `-r RESOURCE, --resource RESOURCE` |   | 请参考原文 |
| `--uac-admin` |   | 请参考原文 |
| `--uac-uiaccess` |   | 请参考原文 |

## `.spec`文件打包

### `--onefile`打包
- 生成 `.spec` 文件：`pyinstaller -F xxx.py`
- 编写 `.spec` 内容
- 打包：`pyinstaller xxx.py`

```text
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['stock_main.py'],  # 此列表存放项目设计的所有Python脚本文件
    pathex=["/upper_computer/src/upper_computer_ui/script/qs_apis"], # 此列表为项目的绝对路径
    binaries=[],
    # 需要检查代码中是否有__import__或importlib的使用，导致隐式依赖
    datas=[("./config.ini","."), ("./pyinstaller_data/mini_racer.dll", "."), ("./static","./static"),
    ("./static/swagger-ui","./static/swagger-ui"), ("./data/*","./data"), ("./pyinstaller_data/libmini_racer.glibc.so", ".")],
    hiddenimports=["akshare.data", "talib.stream", "stock_main"],
    hookspath=["./pyinstaller_data"],  # 指定hook目录
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='stock_main',  # 打包程序的名字
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,       # 程序运行时是否打开控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

### `--onedir`打包

- 生成 `.spec` 文件：`pyinstaller -D xxx.py`
- 编写 `.spec` 内容
- 打包：`pyinstaller xxx.py`
- 区别：注意跟`--onefile`打包的区别，`--onedir`打包会生成一个文件夹，里面包含所有打包的文件，而`--onefile`打包会生成一个exe文件

```text
# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
```

## 常见打包错误及解决办法

### 1、在用`pyinstaller`打包（-F 选项），如果用到的第三方库含有`data`文件，而`pyinstaller`又没有自带该第三方库文件的`hook`的时候，执行打包后的`exe`一般会报以下错误

> [友情链接](https://blog.csdn.net/lucyTheSlayer/article/details/92795220)

```text
FileNotFoundError: [Errno 2] No such file or directory: ‘C:\Users\ADMINI~1\AppData\Local\Temp\1\_MEI54762\jieba\dict.txt’
[20784] Failed to execute script bat_server
```

上面就是没把`python`库`jieba`的`dict.txt`打包进来，导致了错误。

那么，解决问题也很简单，自己写个`hook`，然后放进`pyinstaller`的`hooks`里面即可。

`hook`文件的命名规范为: `hook-【库名】.py`，以结巴分词为例，即为`hook-jieba.py`，然后简单敲入以下两行：

```python
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

datas = copy_metadata('jieba')  # 解决 `pkg_resources` 错误
datas.extend(collect_data_files("jieba"))  # 解决静态文件不存在错误
```

接下来，找到`pyinstaller`的`hooks`文件夹，大概位于：
`python`根目录`\Lib\site-packages\PyInstaller\hooks`下，然后把`hook-jieba.py`丢进去
> 注意是`\Lib\site-packages\PyInstaller\hooks` 不是 `\Lib\site-packages\PyInstaller\utils\hooks`
>
> **或者可以使用参数 `--additional-hooks-dir HOOKSPATH` 指定用户自定义的 hook 文件夹目录**

最后，回到项目根目录，用`pyinstaller`打包即可。（注意需要把`build`目录删了，使`pyinstaller`从头开始打包）

当看到`pyinstaller`的日志里使用了我们自定义的`hook`后，就万事大吉了。

ok

> 打包tushare 或 akshare 的时候也有类似问题，下次可以直接用此法解决。

### 2、打包成一个文件后，文件太大，运行启动慢

- 1、临时文件解压耗时（`--onefile`模式的核心问题）

    `PyInstaller` 的 `--onefile` 模式会将所有依赖打包成一个可执行文件，启动时需解压到临时目录（如 `/tmp` 或 `%TEMP%`）。过大的文件解压会导致大量磁盘 `I/O`，表现为频繁的 `read/write` 系统调用。

  - 验证方法：
    - 启动程序时观察临时目录（`sys._MEIPASS`）是否生成大量文件。
    - 使用工具监控磁盘 `I/O`（如 `iotop、Windows` 资源监视器）。
  - 解决方案：
    - 避免使用 `--onefile` 模式：改用 `--onedir` 生成目录结构，避免解压开销。
    - 精简依赖：移除未使用的库（如 `pandas、numpy、PyQt` 等）。
    - `UPX` 压缩：使用 `--upx-dir` 压缩二进制依赖（需安装 `UPX`）,注意：`Linux` 下不能使用。
- 2、隐式依赖和冗余库
`PyInstaller` 可能误打包未使用的库（尤其是动态导入的模块），导致体积膨胀。
  - 验证方法：分析生成的可执行文件中的 `PYZ` 内容：`pyi-archive_viewer your_program.exe`
    - 检查是否有 `tensorflow、torch` 等大型库被误打包。
  - 解决方案：
    - 手动排除冗余库：在 `.spec` 文件中添加 `excludes`：

      ```python
      a = Analysis(
          ['your_script.py'],
          excludes=['tensorflow', 'cv2', 'scipy'],
          ...
      )
      ```

    - 使用 `--exclude-module`：`pyinstaller --exclude-module tensorflow your_script.py`

### 3、使用`--add-data`或在`spec`文件的`datas`添加外部文件，在程序内路径怎么修改？

- 获取资源的运行时路径

```python
import sys
import os

def resource_path(relative_path):
    """ 获取资源的绝对路径（兼容开发环境和打包后的环境） """
    if hasattr(sys, '_MEIPASS'):
        # 打包后的资源路径：sys._MEIPASS + relative_path
        base_path = sys._MEIPASS
    else:
        # 开发环境的资源路径：当前脚本所在目录 + relative_path
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
```

- `spec`文件

```python
# your_script.spec
a = Analysis(
    ['your_script.py'],
    datas=[
        ('src/config/settings.ini', 'config'),  # 将 src/config/settings.ini 打包到 config 目录
        ('images/*.png', 'images')             # 将 images 目录下所有 PNG 文件打包到 images 目录
    ],
    ...
)
```

- 在代码中使用资源

```python
# 读取配置文件
config_path = resource_path("config/settings.ini")
with open(config_path, "r") as f:
    content = f.read()

# 加载图片
image_path = resource_path("images/logo.png")
Image.open(image_path).show()
```

## pyinstxtractor 反编译

- `pyinstxtractor`[下载地址](https://link.zhihu.com/?target=https%3A//github.com/extremecoders-re/pyinstxtractor/blob/master/pyinstxtractor.py)，不要跟其他教程下载旧的，至少不适用`python3.9`
- 跟你的 `exe` 放到一个文件夹
- 执行`python pyinstxtractor.py myapp.exe`
- 生成一个 `myapp.exe_extracted` 的文件夹
- 安装`uncompyle6`，直接`pip`就行`pip install uncompyle6`
- `uncompyle6 myapp.pyc > myapp.py`
###### datetime:2022-01-08 14:34:00

###### author:nzb

## Linux使用注意事项（新手必看）

### Linux 严格区分大小写

* 和 Windows 不同，Linux 是严格区分大小写的，包括文件名和目录名、命令、命令选项、配置文件设置选项等。

* 例如，Windows 系统桌面上有一个名为 Demo 的文件夹，当我们在桌面上再新建一个名为 demo 的文件夹时，系统会提示文件夹命名冲突；而 Linux 系统不会，Linux 系统认为 Demo 文件和 demo
  文件不是同一个文件，因此在 Linux 系统中，Demo 文件和 demo 文件可以位于同一目录下。

* 因此，初学者在操作 Linux 系统时要注意区分大小写的不同。

### Linux 中所有内容（包括硬件设备）以文件形式保存

Linux 中所有内容都是以文件的形式保存和管理的（硬件设备也是文件），这和 Windows 完全不同，Windows 是通过设备管理器来管理硬件的。比如说，Linux 的设备文件保存在 /dev/ 目录中，硬盘文件是
/dev/sd[a-p]，光盘文件是 /dev/hdc 等。

### Linux 不靠扩展名区分文件类型

* 我们都知道，Windows 是依赖扩展名区分文件类型的，比如，".txt" 是文本文件、".exe" 是执行文件、".ini" 是配置文件、".mp4" 是小电影等。但 Linux 不是。

* Linux 系统通过权限位标识来确定文件类型，且文件类型的种类也不像 Windows 下那么多，常见的文件类型只有普通文件、目录、链接文件、块设备文件、字符设备文件等几种。Linux 的可执行文件不过就是普通文件被赋予了可执行权限而已。

* Linux 中的一些特殊文件还是要求写 "扩展名" 的，但大家小心，并不是 Linux 一定要靠扩展名来识别文件类型，写这些扩展名是为了帮助管理员来区分不同的文件类型。这样的文件扩展名主要有以下几种：

    * 压缩包：Linux 下常见的压缩文件名有 *.gz、*.bz2、*.zip、*.tar.gz、*.tar.bz2、*.tgz
      等。为什么压缩包一定要写扩展名呢？很简单，如果不写清楚扩展名，那么管理员不容易判断压缩包的格式，虽然有命令可以帮助判断，但是直观一点更加方便。另外，就算没写扩展名，在 Linux 中一样可以解压缩，不影响使用。

    * 二进制软件包：CentOS 中所使用的二进制安装包是 RPM 包，所有的 RPM 包都用".rpm"扩展名结尾，目的同样是让管理员一目了然。

    * 程序文件：Shell 脚本一般用 "*.sh" 扩展名结尾，其他还有用 "*.c" 扩展名结尾的 C 语言文件等。

    * 网页文件：网页文件一般使用 "*.php" 等结尾，不过这是网页服务器的要求，而不是 Linux 的要求。

    * 在此不一一列举了，还有如日常使用较多的图片文件、视频文件、Office 文件等，也是如此。

### Linux中所有存储设备都必须在挂载之后才能使用

* Linux 中所有的存储设备都有自己的设备文件名，这些设备文件必须在挂载之后才能使用，包括硬盘、U 盘和光盘。

* 挂载其实就是给这些存储设备分配盘符，只不过 Windows 中的盘符用英文字母表示，而 Linux 中的盘符则是一个已经建立的空目录。我们把这些空目录叫作挂载点（可以理解为 Windows 的盘符），把设备文件（如
  /dev/sdb）和挂载点（已经建立的空目录）连接的过程叫作挂载。这个过程是通过挂载命令实现的，具体的挂载命令后续会讲。

### Windows 下的程序不能直接在 Linux 中使用

* Linux 和 Windows 是不同的操作系统，两者的安装软件不能混用。例如，Windows 系统上的 QQ 软件安装包无法直接放到 Linux 上使用。

* 系统之间存在的这一差异，有弊也有利。弊端很明显，就是所有的软件要想安装在 Linux 系统上，必须单独开发针对 Linux 系统的版本（也可以依赖模拟器软件运行）；好处则是能感染 Windows 系统的病毒（或木马）对 Linux
  无效。
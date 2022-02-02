###### datetime:2022-01-08 14:34:00

###### author:nzb

## Linux系统安装

### VMware虚拟机安装

* VMware官网下载，不推荐安装双系统

*  特点：

    * 不需要分区就能在物理机上使用两种以上的操作系统

    * 物理机和虚拟机能实现网络通信

    * 可以设定并随时修改虚拟机操作系统的硬件环境

* 要求：

    * CPU：主频1GHz以上

    * 内存：1GB以上

    * 硬盘：8GB以上

* 镜像下载：[官网下载](https://www.centos.org/download/mirrors/):
  几个版本：DVD版本，Everything版本，minimal版本，LiveGnome版本，KdeLive版本，livecd版本，NetInstall版本

### 系统分区

* 主分区：最多只能有4个

*  扩展分区：最多只能有一个；主分区加扩展分区最多有4个；不能写入数据，只能包含逻辑分区

*  逻辑分区：可以和主分区一样正确的写入数据和格式化

*  注意：兄弟连这套视频录制时间较为久远，当时的硬盘分区形式是MBR的，所以上述的分区限制也只 是针对MBR分区形式，对于GPT分区形式而言，则没有上述限制了。

*  电脑根据主板的不同（BOIS或者UEFI），会决定硬盘选择MBR分区方案还是GPT分区方案：

    * BIOS + MBR

    * UEFI + GPT

    * 两者区别：

        * 也就是说，电脑使用传统BIOS主板，建议使用MBR分区方案；电脑使用UEFI主板，建议使用GPT分区方案

        * MBR分区表最多只能识别2TB左右的空间，大于2TB的容量将无法识别从而导致硬盘空间浪费；GPT分区表则能够识别2TB以上的硬盘空间。

        * MBR分区表最多只能支持4个主分区或三个主分区+1个扩展分区(逻辑分区不限制)；GPT分区表在Windows系统下可以支持128个主分区。

        * 在MBR中，分区表的大小是固定的；在GPT分区表头中可自定义分区数量的最大值，也就是说GPT分区表的大小不是固定的。

* 硬盘分区的作用： 把一块大硬盘分成几块

* 格式化的作用： 写入文件系统（1.把硬盘分成一个个等大小的数据块 同时2.建立一个inode列表）

* Linux中的所有硬件都是文件：

    * 硬盘文件名：

        * IDE硬盘：/dev/hd[a-d]

        * SCSI/SATA/USB硬盘：/dev/sd[a-p]

        * 光驱：/dev/cdrom或/dev/sr0

        * 鼠标：/dev/mouse

    * 分区文件名：

        * /dev/hda[数字]

        * /dev/sda[数字]

* 挂载： 给分区分配挂载点

    * /根分区

    * swap交换分区（内存两倍，最大不超多2GB）

    * /boot启动分区（200MB足够）

* 总结：

    * 分区：把大硬盘分为小的分区

    * 格式化：写入文件系统，同时会清空数据

    * 分区设备文件名：给每个分区定义设备文件名

    * 挂在：给每个分区分配挂载点，这个挂在点必须是空目录

### Linux系统安装

把镜像加进去，点击启动，然后用图形界面配置分区和其他的自定义选项，确定定义root用户的密码和普通用户的账号和密码。然后等待安装完成即可。

### 远程登陆管理工具

* 三种网络连接方式：

    * 桥接模式：虚拟机使用物理网卡

    * NAT模式：虚拟机使用vmnet8虚拟网卡

    * Host-only模式：虚拟机使用vmnet1虚拟网卡，并且只能和本机通信

* 临时配置ip：ifconfig ens33 192.168.XXX.XXX

* 永久配置ip：
    - 查看网络接口：ifconfig
    - 去网络接口的配置文件进行修改
        ```bash
        [root@bogon ~]# vim /etc/sysconfig/network-scripts/ifcfg-ens33/  ens33是网卡接口
        ```
    - 配置文件
        ```text
        TYPE=“Ethernet”
        PROXY_METHOD=“none”
        BROWSER_ONLY=“no”
        BOOTPROTO=“none” //dhcp是自动获取
        DEFROUTE=“yes”
        IPV4_FAILURE_FATAL=“no”
        IPV6INIT=“yes”
        IPV6_AUTOCONF=“yes”
        IPV6_DEFROUTE=“yes”
        IPV6_FAILURE_FATAL=“no”
        IPV6_ADDR_GEN_MODE=“stable-privacy”
        NAME=“ens33”
        UUID=“d8ee940a-1a27-4417-9ae8-88a5364ee4d1”
        DEVICE=“ens33”
        ONBOOT=“yes” //引导激活
        IPADDR=172.16.10.188 //ip地址
        NETMASK=255.255.255.0 //子网掩码
        GATEWAY=172.16.10.254 //网关
        DNS1=222.88.88.88 //DNS
        ```
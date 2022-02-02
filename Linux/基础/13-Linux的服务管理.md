###### datetime:2022-01-08 14:34:00

###### author:nzb

## Linux的服务管理

### 服务简介与分类

* Linux服务分类

    * RPM包默认安装的服务

        * 独立的服务

        * 基于xinetd（超级守护进程）服务

    * 源码包安装的服务（第三方源码包）

* 服务的启动与自启动

    * 查询已安装的服务

        * RPM包安装的服务：chkconfig --list

            * 查看RPM包安装的服务按照运行级别的自启动状态。

              `查看是否在系统下次启动时自启动，而不是查看服务是否当前已启动`

            * 查询当前启动的服务

                * ps aux

                * netstat

                * service --status-all

        * 源码包安装的服务：没有命令，只能去服务安装位置查看，一般在/usr/local/下

    * 其实源码包和RPM包安装的服务在Linux中的区别就是安装位置不同

        * 源码包安装在指定位置，一般是/usr/local/下

        * RPM包安装在默认位置中，配置文件在/etc/下，启动命令在/etc/rc.d/init.d/下，分散到很多文件夹下

### RPM包安装服务的管理

* RPM包安装的服务默认保存位置：(特殊文件有自己的默认保存位置)

  ![](../res/76f488bd-43bd-4146-a556-fd93f9e1d680-5771924.jpg)

* 独立服务的管理

    * 启动方式

        * /etc/init.d/ 独立服务名 start | stop | status | restart

          `推荐使用`

        * service 独立服务名 start | stop | restart | status

          `rea hat独有，service 相当于/etc/inin.d`

    * 自启动方式

        * 方式一：打开自启动：chkconfig [--level 运行级别] [独立服务名] on

          `不支持源码包安装的服务`

            * 关闭自启动：chkconfig 独立服务名 off

            * 默认就是：2345

                * ![](../res/e03e80d7-bd16-46ba-928b-f081454a6714-5771924.jpg)

        * 方式二：修改/etc/rc.d/rc.local文件，加入需要自启动的服务名

          `推荐`

        * 方式三：使用ntsysv命令管理自启动，图形界面很直观

          `red hat专有`

* 基于xinetd(超级守护进程)服务的管理

    * 默认情况下Linux是没有xinted的，需要手动安装yum -y install xinetd

    * 然后用chkconfig --list查看，基于xinetd的服务不占用内存，但是需要的响应时间更长

    * 基于xinetd的服务的启动，修改/etc/xinetd.d/下对应的服务的配置文件,然后service xinetd restart

    * 基于xinetd的服务的自启动：

        * chkconfig 服务名 on和chkconfig 服务名 off

        * 图形界面工具：ntsysv

    * 基于xinetd的启动和自启动是通用的，两者区分不是很严格，这种设置不利于管理，所以现在基于xinetd的服务越来越少了

      `自启动关闭，服务也会关闭，2者相通`

### 源码包安装服务的管理

* 源码包安装的服务默认保存位置：/usr/local/

* 源码包安装服务的启动和关闭(用绝对路径的启动脚本启动)：/usr/local/apache2/bin/apachectl start|stop

* 一般每一个源码包都有安装说明INSTALL，应该查看里面的启动方法

* 源码包安装服务的自启动：

    * vim /etc/rc.d/rc.local加入/usr/local/apache2/bin/apachectl start

* 把源码包服务的启动脚本软连接到/etc/init.d/目录下和chkconfig --add 服务名，就可以实现service，chkconfig和ntsysv命令管理源码包安装服务，但是并不推荐，容易混乱。

### 总结

![](../res/aff76657-a46d-46b7-bc75-712860dcfbe4-5771924.jpg)
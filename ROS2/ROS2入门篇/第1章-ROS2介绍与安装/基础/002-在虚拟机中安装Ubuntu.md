###### datetime:2023/09/11 14:16

###### author:nzb

> 该项目来源于[大佬小鱼的动手学ROS2](https://fishros.com/d2lros2)

# 2.在虚拟机中安装Ubuntu

为方便学习，可以先使用虚拟机在Windows上使用Ubuntu，当然除了虚拟机还有Windows子系统等方法，你可以自行尝试。

## 1.下载

所谓虚拟机，就是在你的电脑已有的系统上再使用软件模拟出另外一个系统。比较著名的软件就是Vmware了，因为Vmware是收费的，我们使用他们的非商业版本Vmware-Player。

### 1.1下载Vmvare

Vmvare官方下载链接：[VMware Workstation Player - My VMware](https://my.vmware.com/en/web/vmware/downloads/details?downloadGroup=WKST-PLAYER-1612&productId=1039&rPId=66621)

大家在浏览器里打开网页，然后点击下图中的DownLoad Now即可，注意上面一个是windows版本，下面一个是linux版本的。

![image-20210719182446728]（imgs/image-20210719182446728.png)

### 1.1下载ubuntu

下载好后虚拟机安装包后，接下来下载ubuntu镜像文件。

下载地址：[Index of /ubuntu-releases/22.04/ (ustc.edu.cn)](http://mirrors.ustc.edu.cn/ubuntu-releases/22.04/)，点开上面的网址，你会看到下面的页面：

![image-20220526132619902]（imgs/image-20220526132619902.png)



话不多说点这个：`ubuntu-22.04-desktop-amd64.iso` 下载， 你可能会有疑问，为啥是amd64，因为amd64位的架构是目前最流行的。

下载好后，你应该得到这两个文件：

```
ubuntu-22.04-desktop-amd64.iso
VMware-player-full-16.2.3-19376536
```

## 2.安装Vmware

双击执行VmWarePlayer，等待一下，然后一路next。

![image-20210719185440656](imgs/image-20210719185440656.png)

![image-20210719185458219](imgs/image-20210719185458219.png)

![image-20210719185545384](imgs/image-20210719185545384.png)

![image-20210719185556305](imgs/image-20210719185556305.png)

![image-20210719185604490](imgs/image-20210719185604490.png)

![image-20210719185613586](imgs/image-20210719185613586.png)

点击安装，等待一下下

![image-20210719185625695](imgs/image-20210719185625695.png)

点完成

![image-20210719185706443](imgs/image-20210719185706443.png)

此时桌面上应该看到对应的图标了。

![image-20210719185758076](imgs/image-20210719185758076.png)

双击打开：

![image-20210719185817378](imgs/image-20210719185817378.png)

肯定免费白嫖啦，点继续

![image-20210719185845413](imgs/image-20210719185845413.png)

点完成，此时主界面就出来了

![image-20210719185912397](imgs/image-20210719185912397.png)

## 3.安装Ubuntu22虚拟机

点开文件新建虚拟机



![image-20210719185950677](imgs/image-20210719185950677.png)

点开后选第二个选项

![image-20210719190042507](imgs/image-20210719190042507.png)

然后点浏览，找到我们下载的ubuntu镜像，点击打开，接着点击下一步

![image-20220526133220329](imgs/image-20220526133220329.png)

输入一下信息，名称用了ROS2，密码用的是123

![image-20210719191648124](imgs/image-20210719191648124.png)

**点下一步，点浏览，更改一下位置**

![image-20210719191817025](imgs/image-20210719191817025.png)

点下一步，然后改一下磁盘大小，改成80G。

![image-20210719191850033](imgs/image-20210719191850033.png)

然后点下一步

![image-20210719191919143](imgs/image-20210719191919143.png)

这里大家可以根据自己电脑自定义一下，比如本机有16G的内存，8核CPU，这里分给虚拟机4核8G。

![image-20210719192023103](imgs/image-20210719192023103.png)

然后点击完成

![image-20210719192050412](imgs/image-20210719192050412.png)

看到类似上面这个界面，不要着急，保持耐心，等待即可。

最终装好了之后，你就可以看到登录界面，输入密码即可进入系统。

![image-20220526133606266](imgs/image-20220526133606266.png)

## 4.更改分辨率

在桌面空白处右击，选择Display Settings

![image-20220526133831055](imgs/image-20220526133831055.png)

修改Resolution

![image-20220526133910318](imgs/image-20220526133910318.png)

右上角点Apply即可修改分辨率

![image-20220526134017759](imgs/image-20220526134017759.png)

--------------
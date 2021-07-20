###### datetime:2019/6/14 9:04
###### author:nzb

> 笔记格式：`{ { } }` 写成 `{ { } }` ， `{ % % }` 写成 `{ %  % }`，空格隔开
> 
> [gitbook 生成文档](https://github.com/bluesnie/git_test)
> 
> 注意：SUMMARY.md 目录中有的 markdown 才会转成静态 html, 并且才能全局搜索得到

# [电子书](https://bluesnie.github.io/Learning-notes/)

# 学习笔记

## [算法](./Algorithm)

- [算法图解](./Algorithm/算法图解.pdf)

## [Linux](./Linux)

- [入门基础](./Linux/基础/Linux.html)

## [Python](./Python)

### [Python语言基础](./Python/Python语言基础/)

- [Python PEP8编码规范 ](./Python/PEP8.md) 
| [初识Python](./Python/Python语言基础/00-初识Python.md)
| [语言元素](./Python/Python语言基础/01-语言元素.md)
- [分支结构](./Python/Python语言基础/02-分支结构.md)
| [循环结构](./Python/Python语言基础/03-循环结构.md)
| [构造程序逻辑](./Python/Python语言基础/04-练习.md)
| [函数和模块的使用](./Python/Python语言基础/05-函数和模块的使用.md)
- [字符串和常用数据结构](./Python/Python语言基础/06-字符串和常用数据结构.md)
| [面向对象编程基础](./Python/Python语言基础/07-面向对象编程基础.md)
| [面向对象编程进阶](./Python/Python语言基础/08-面向对象编程进阶.md)
- [图形用户界面和游戏开发](./Python/Python语言基础/09-图形用户界面和游戏开发.md)
| [文件和异常](./Python/Python语言基础/10-文件和异常.md)
| [字符串和正则表达式](./Python/Python语言基础/11-字符串和正则表达式.md)
- [进程和线程](./Python/Python语言基础/12-进程和线程.md)
| [网络编程入门](./Python/Python语言基础/13-网络编程入门.md)
| [网络应用开发](./Python/Python语言基础/14-网络应用开发.md)
| [图像和文档处理](./Python/Python语言基础/15-图像和办公文档处理.md)

### [Python语言进阶 ](./Python/Python语言进阶)

- [常用数据结构](./Python/Python语言进阶/01-常用数据结构和算法.md)
- [函数的高级用法](./Python/Python语言进阶/02-函数的高级用法.md) - “一等公民” / 高阶函数 / Lambda函数 / 作用域和闭包 / 装饰器
- [面向对象高级知识](./Python/Python语言进阶/03-面向对象高级知识.md) - “三大支柱” / 类与类之间的关系 / 垃圾回收 / 魔术属性和方法 / 混入 / 元类 / 面向对象设计原则 / GoF设计模式
- [迭代器和生成器](./Python/Python语言进阶/04-迭代器和生成器.md) - 相关魔术方法 / 创建生成器的两种方式 / 
- [并发和异步编程](./Python/Python语言进阶/05-并发和异步编程.md) - 多线程 / 多进程 / 异步IO / async和await

### [Python 第三方库](./Python/第三方库)

#### [Django](./Python/第三方库/Django)

- [快速上手](./Python/第三方库/Django/01-快速上手.md)
| [深入模型](./Python/第三方库/Django/02-深入模型.md)
| [静态资源和Ajax请求](./Python/第三方库/Django/03-静态资源和Ajax请求.md)
| [Django模板系统](./Python/第三方库/Django/04-Django模板系统.md)
- [Django的View(视图)](./Python/第三方库/Django/05-Django的View.md)
| [Django的路由系统](./Python/第三方库/Django/06-Django的路由系统.md)
| [Django ORM相关操作](./Python/第三方库/Django/07-Django-ORM相关操作.md)
- [Cookie、Session和分页](./Python/第三方库/Django/08-Cookie、Session和分页.md)
| [Form、ModelForm组件](./Python/第三方库/Django/09-Form和ModelForm组件.md)
| [中间件](./Python/第三方库/Django/10-中间件.md)

#### [Django-REST-framework](./Python/第三方库/Django/Django-REST-framework.md)

- [Django生命周期](./Python/第三方库/Django/Django-REST-framework.md#Django生命周期:(rest_framework))
| [Django中间件](./Python/第三方库/Django/Django-REST-framework.md#中间件)
- Django-Rest-framework组件:
    - [认证](./Python/第三方库/Django/Django-REST-framework.md#一认证)
    | [权限](./Python/第三方库/Django/Django-REST-framework.md#二权限)
    | [频率控制(节流)](./Python/第三方库/Django/Django-REST-framework.md#三频率控制节流)
    | [版本(全局配置就行)](./Python/第三方库/Django/Django-REST-framework.md#四版本全局配置就行)
    | [解析器(全局配置就行)](./Python/第三方库/Django/Django-REST-framework.md#五解析器全局配置就行)
    - [序列化](./Python/第三方库/Django/Django-REST-framework.md#六序列化)
    | [分页](./Python/第三方库/Django/Django-REST-framework.md#七分页)
    | [视图](./Python/第三方库/Django/Django-REST-framework.md#八视图)
    | [路由](./Python/第三方库/Django/Django-REST-framework.md#九路由)
    | [渲染器](./Python/第三方库/Django/Django-REST-framework.md#十渲染器)
    | [django组件：content-type](./Python/第三方库/Django/Django-REST-framework.md#十一django组件content_type)
      

#### [Django项目开发经验](./Python/第三方库/Django/Django开发经验)

- [登录相关](./Python/第三方库/Django/Django开发经验/02-Django-restframework登录相关.md)
| [异常处理手柄](./Python/第三方库/Django/Django开发经验/01-Django-restframework重写异常处理手柄.md)
| [过滤相关](./Python/第三方库/Django/Django开发经验/03-Django-restframework过滤类相关.md)
| [存储类重写](./Python/第三方库/Django/Django开发经验/04-Django-Fastdfs重写存储类.md)
| [序列化相关](./Python/第三方库/Django/Django开发经验/05-Django-restframework序列化相关.md)
| [自动化测试](./Python/第三方库/Django/Django开发经验/06-api接口自动化测试.md)
| [接口加速缓存](./Python/第三方库/Django/Django开发经验/07-为接口加速加缓存.md)

#### [PyQt5](./Python/第三方库/PyQt5)
- [导航](./Python/第三方库/PyQt5/README.md)
- 窗口 / 按钮 / 垂直布局和水平布局 / 栅格布局 / 布局添加标签和背景图
- 单选框 / 复选框 / 键盘提示 / 行编辑器 / 按钮组 / 布局组 / 无边框窗口
-  框架 / 分离器 / 滑动条 / 滚动条 / 刻度盘 / 上下拨号 / 生成随机数 
- 进度条 / 工具框 / 菜单栏工具栏 / 文档编辑框 / 字体文本框 / 颜色文本框
- 打印 / 打印预览 / 打印PDF / 消息提示框 / 右键菜单 / 选项卡 / 堆叠小部件
- 可停靠的窗口小部件 / 日历 / 单选下拉框 / 首字母模糊填充 / 打开更多窗口
- 时间编辑 / 列表部件 / 

## [数据库基础和进阶](./Database)

### [关系型数据库MySQL](./Database/MySQL)
  - 关系型数据库概述
  - MySQL的安装和使用
  - SQL的使用
    - DDL - 数据定义语言 - create / drop / alter
    - DML - 数据操作语言 - insert / delete / update / select
    - DCL - 数据控制语言 - grant / revoke
  - 相关知识
    - 范式理论 - 设计二维表的指导思想
    - 数据完整性
    - 数据一致性
  - 在Python中操作MySQL
  - [计算机二级](./Database/MySQL/计算机二级MySQL.md)
  | [数据库三大范式](./Database/MySQL/数据库三大范式.md)
  | [MySQL主从复制](./Database/MySQL/主从复制.md)
  

### [NoSQL入门](./Day36-40/39.NoSQL入门.md)
  - NoSQL概述
  | Redis概述
  | Mongo概述

#### Redeis
  - [Redis安装与配置文件](./Database/NoSQL/02-Redis安装与配置文件.md)
  - [Redis基础](./Database/NoSQL/03-Redis基础.md)
    - [Redis键](./Database/NoSQL/03-Redis基础.md#redis键key) 
    | [Redis字符串](./Database/NoSQL/03-Redis基础.md#redis-字符串string) 
    | [Redis哈希](./Database/NoSQL/03-Redis基础.md#redis-哈希hash)
    | [Redis列表](./Database/NoSQL/03-Redis基础.md#redis-列表list)
    | [Redis集合](./Database/NoSQL/03-Redis基础.md#redis-集合set)
    | [Redis有序集合](./Database/NoSQL/03-Redis基础.md#redis-有序集合sorted-set)
    | [Redis发布订阅](./Database/NoSQL/03-Redis基础.md#redis-发布订阅)
    - [Redis多数据库](./Database/NoSQL/03-Redis基础.md#redis多数据库)
    | [Redis事务](./Database/NoSQL/03-Redis基础.md#redis-事务)
    | [Redis数据淘汰策略](./Database/NoSQL/03-Redis基础.md#redis数据淘汰策略redisconf)
    | [Redis持久化](./Database/NoSQL/03-Redis基础.md#redis持久化)
    | [Redis缓存与数据库Mysql一致性方案](./Database/NoSQL/03-Redis基础.md#redis缓存与数据库mysql一致性方案)

## [数据分析](./MachineLearning/)

### [数学基础](./MachineLearning/数学基础)
    
- [高等数学](./MachineLearning/数学基础/高等数学.pdf)
| [概率论](./MachineLearning/数学基础/概率论.pdf)
| [微积分](./MachineLearning/数学基础/微积分.pdf)
| [SVD](./MachineLearning/数学基础/SVD.pdf)
| [似然函数](./MachineLearning/数学基础/似然函数.pdf)
| [后验概率估计](./MachineLearning/数学基础/后验概率估计.pdf)
| [拉格朗日乘子法](./MachineLearning/数学基础/拉格朗日乘子法.pdf)
| [核函数](./MachineLearning/数学基础/核函数.pdf)
- [梯度](./MachineLearning/数学基础/梯度.pdf)
| [概率分布与概率密度](./MachineLearning/数学基础/概率分布与概率密度.pdf)
| [泰勒公式](./MachineLearning/数学基础/泰勒公式.pdf)
| [激活函数](./MachineLearning/数学基础/激活函数.pdf)
| [熵](./MachineLearning/数学基础/熵.pdf)
| [特征值与特征向量](./MachineLearning/数学基础/特征值与特征向量.pdf)
| [矩阵](./MachineLearning/数学基础/矩阵.pdf)

### [数据分析库](./MachineLearning/MatplotlibNumpyPandas)
    
- [numpy基础](./MachineLearning/MatplotlibNumpyPandas/numpy基础.md)
| [pandas基础](./MachineLearning/MatplotlibNumpyPandas/Pandas快速入门.md)
| [matplotlib基础](./MachineLearning/MatplotlibNumpyPandas/matplotlib基础.md)

## [Web前端](./Web/)

- [Web前端入门](./Web/Web前端概述.md)
| [HTML5](./Web/HTML5)
| [CSS](./Web/CSS)
| [JavaScript](./Web/JavaScript)
- [jQuery](./Web/JavaScript/框架)
| [jQuery-UI](./Web/JavaScript/框架/jQuery-UI.md)
| [Vue.js](./Web/JavaScript/框架)
| [RESTful规范](./Web/RESTful.md)

## [工具](./Utils/)

### [Docker](./Utils/docker/Docker.md)

- 基本组成 / 容器相关技术简介 / 客户端与守护进程 / 守护进程的配置和操作
- 远程访问 / 容器 / 容器中部署静态网站 / 查看和删除镜像 / 获取和推送镜像
- 构建镜像 / DockerFile指令 / 构建过程 / 容器的网络连接 / 容器的数据卷
- 数据卷容器 / 数据卷的备份和还原
- [Docker入门PDF](./Utils/docker/Docker入门教程.pdf)
- [Docker部署Django Uwsgi+Nginx+MySQL+Redis](./Utils/docker/django_demo_docker/README.md)
- [Docker简单部署Django的FastDFS](./Utils/docker/FastDFS.md)

### [Git](./Utils/Git基本命令.md)

- 帮助信息 / git配置(全局配置) / 初始化项目 / 查看状态 / 添加文件 / 提交文件 / 查看提交日志
- 查看文件修改前后的区别 / git跟踪rename文件和移动文件 / 删除文件 / 恢复文件 / 恢复提交 
- 重置提交指针 / 查看、创建、切换分支 / 查看两个分支的区别 / 合并分支 / 解决合并冲突 / 重命名和删除分支
- 保存修改进度 / 添加别名 / 全局忽略跟踪文件 / 项目级忽略文件 / 创建远程版本库 / 推送版本库
- 修改远程仓库地址 / 克隆版本库到本地 / 更新本地版本库 / 基于版本库开发自己的版本库 / 添加pull request / 添加贡献者

### [Nginx](./Utils/Nginx.md)

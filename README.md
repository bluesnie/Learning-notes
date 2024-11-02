###### datetime:2019/6/14 9:04

###### author:nzb

> 笔记格式：`{ { } }` 写成 `{ { } }` ， `{ % % }` 写成 `{ % % }`，空格隔开
>
> 数学公式可以先用一个 `$`编写，`vscode`可以显示，然后再替换成两个`$$`，生成的文档才能显示
>
> [gitbook 生成文档](https://github.com/bluesnie/git_test)
>
> 注意：SUMMARY.md 目录中有的 markdown 才会转成静态 html, 并且才能全局搜索得到

# [电子书](https://bluesnie.github.io/Learning-notes/)

# 学习笔记

## [Linux](./Linux)

- [入门基础](./Linux/Linux.md)
    - [Linux系统简介](Linux/基础/01-Linux系统简介.md)
    - [Linux系统安装](Linux/基础/02-Linux系统安装.md)
    - [Linux使用注意事项](Linux/基础/03-Linux使用注意事项-新手必看.md)
    - [Linux各目录的作用](Linux/基础/04-Linux各目录的作用.md)
    - [服务器注意事项](Linux/基础/05-服务器注意事项.md)
    - [Linux常用命令](Linux/基础/06-Linux常用命令.md)
    - [强大的文本编辑器Vim](Linux/基础/07-强大的文本编辑器Vim.md)
    - [Linux软件包管理](Linux/基础/08-Linux软件包管理.md)
    - [Linux中的用户管理](Linux/基础/09-Linux中的用户管理.md)
    - [Linux权限管理](Linux/基础/10-Linux权限管理.md)
    - [Linux的文件系统管理](Linux/基础/11-Linux的文件系统管理.md)
    - [Shell](Linux/基础/12-Shell.md)
    - [Linux的服务管理](Linux/基础/13-Linux的服务管理.md)
    - [Linux系统管理](Linux/基础/14-Linux系统管理.md)
    - [Linux日志管理](Linux/基础/15-Linux日志管理.md)
    - [Linux启动管理](Linux/基础/16-Linux启动管理.md)
    - [Linux备份与恢复](Linux/基础/17-Linux备份与恢复.md)

- [命令指北](./Linux/命令指北)
    - [strace](Linux/命令指北/01-strace.md)
    - [ps和pstree](Linux/命令指北/02-ps和pstree.md)
    - [top](Linux/命令指北/03-top.md)
    - [py-spy](Linux/命令指北/04-性能分析工具-py-spy.md)
    - [nc-ncat](Linux/命令指北/05-网络工具nc-ncat命令.md)
    - [tc网络流量控制](Linux/命令指北/06-tc网络流量控制.md)
    - [iptables](Linux/命令指北/07-iptables.md)
    - [tmux](Linux/命令指北/08-tmux.md)
    - [ipdb-python调试器](Linux/命令指北/09-ipdb-python调试器.md)

## [Windows](./Windows)

- [程序无响应问题排查](./Windows/program_debug.md)

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
  | [logging日志模块](./Python/Python语言基础/16-logging日志模块.md)
- [单元测试unittest基础](./Python/Python语言基础/unittest/01-单元测试unittest基础.md)
  | [单元测试unittest进阶](./Python/Python语言基础/unittest/02-单元测试unittest进阶.md)
  | [单元测试unittest集成篇](./Python/Python语言基础/unittest/03-单元测试unittest集成篇.md)

### [Python语言进阶 ](./Python/Python语言进阶)

- [常用数据结构](./Python/Python语言进阶/01-常用数据结构和算法.md)
- [函数的高级用法](./Python/Python语言进阶/02-函数的高级用法.md) - “一等公民” / 高阶函数 / Lambda函数 / 作用域和闭包 / 装饰器
- [面向对象高级知识](./Python/Python语言进阶/03-面向对象高级知识.md) - “三大支柱” / 类与类之间的关系 / 垃圾回收 / 魔术属性和方法 / 混入 / 元类 / 面向对象设计原则 / GoF设计模式
- [迭代器和生成器](./Python/Python语言进阶/04-迭代器和生成器.md) - 相关魔术方法 / 创建生成器的两种方式 /
- [并发和异步编程](./Python/Python语言进阶/05-并发和异步编程.md) - 多线程 / 多进程 / 异步IO / async和await
- [asynico异步编程](./Python/Python语言进阶/06-asyncio异步编程.md)
- [GIL全局解释器锁](./Python/Python语言进阶/07-GIL全局解释器锁.md)
- [线程池ThreadPoolExecutor](./Python/Python语言进阶/08-线程池_ThreadPoolExecutor.md)

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

#### [FastAPI](./Python/第三方库/FastAPI)

#### [基础](./Python/第三方库/FastAPI/基础)

- [pydantic](./Python/第三方库/FastAPI/基础/01-pydantic.md)
- [hello_world](./Python/第三方库/FastAPI/基础/02-hello_world.md)
- [请求参数和验证](./Python/第三方库/FastAPI/基础/03-请求参数和验证.md)
- [响应处理和FastAPI配置](./Python/第三方库/FastAPI/基础/04-响应处理和FastAPI配置.md)
- [FastAPI的依赖注入系统](./Python/第三方库/FastAPI/基础/05-FastAPI的依赖注入系统.md)
- [安全、认证和授权](./Python/第三方库/FastAPI/基础/06-安全、认证和授权.md)
- [FastAPI的数据库操作和多应用的目录结构设计](./Python/第三方库/FastAPI/基础/07-FastAPI的数据库操作和多应用的目录结构设计.md)
- [中间件、CORS、后台任务、测试用例](./Python/第三方库/FastAPI/基础/08-中间件、CORS、后台任务、测试用例.md)
- [示例新冠病毒疫情跟踪器API](./Python/第三方库/FastAPI/基础/09-示例新冠病毒疫情跟踪器API.md)
- [apSheduler动态定时任务](./Python/第三方库/FastAPI/基础/10-apSheduler动态定时任务.md)
- [入口文件、全局配置](./Python/第三方库/FastAPI/基础/11-main.md)

#### [PyQt5](./Python/第三方库/PyQt5)

- [导航](./Python/第三方库/PyQt5/README.md)
- 窗口 / 按钮 / 垂直布局和水平布局 / 栅格布局 / 布局添加标签和背景图
- 单选框 / 复选框 / 键盘提示 / 行编辑器 / 按钮组 / 布局组 / 无边框窗口
- 框架 / 分离器 / 滑动条 / 滚动条 / 刻度盘 / 上下拨号 / 生成随机数
- 进度条 / 工具框 / 菜单栏工具栏 / 文档编辑框 / 字体文本框 / 颜色文本框
- 打印 / 打印预览 / 打印PDF / 消息提示框 / 右键菜单 / 选项卡 / 堆叠小部件
- 可停靠的窗口小部件 / 日历 / 单选下拉框 / 首字母模糊填充 / 打开更多窗口
- 时间编辑 / 列表部件 /

#### [PySide](./Python/第三方库/PySide)

- [Qt简介](./Python/第三方库/PySide/01-Qt简介.md)
- [界面设计师QtDesigner](./Python/第三方库/PySide/02-界面设计师QtDesigner.md)
- [发布程序](./Python/第三方库/PySide/03-发布程序.md)
- [常用控件1](./Python/第三方库/PySide/04-常用控件1.md)
- [常用控件2](./Python/第三方库/PySide/05-常用控件2.md)
- [常用控件3](./Python/第三方库/PySide/06-常用控件3.md)
- [常用控件4](./Python/第三方库/PySide/07-常用控件4.md)

#### [OpenCV](./Python/第三方库/OpenCV)

- [图像基本操作](./Python/第三方库/OpenCV/01-图像基本操作.md)
<!-- - [图像处理](./Python/第三方库/OpenCV/02-图像处理.md) -->
- [灰度二值化HSV](./Python/第三方库/OpenCV/02-图像处理-灰度二值化HSV.md)
- [滤波-形态学处理](./Python/第三方库/OpenCV/03-图像处理-滤波-形态学处理.md)
- [图像梯度-边缘检测-图像金字塔](./Python/第三方库/OpenCV/04-图像处理-图像梯度-边缘检测-图像金字塔.md)
- [图像轮廓及绘制图形](./Python/第三方库/OpenCV/05-图像处理-图像轮廓及绘制图形.md)
- [图像模板匹配](./Python/第三方库/OpenCV/06-图像处理-图像模板匹配.md)

#### [Pyinstaller](./Python/第三方库/PyInstaller)

- [pyInstaller打包基础](./Python/第三方库/PyInstaller/01-pyInstaller打包基础.md)

#### [ZeroMQ](./Python/第三方库/ZeroMQ)

- [zmq基础](./Python/第三方库/ZeroMQ/01-zmq基础.md)

## [GoLang](./GoLang)

### [GoLang简明教程](GoLang/Go简明教程)

- [Go语言简明教程](GoLang/Go简明教程/01-Go语言简明教程.md)
- [Gin-简明教程](GoLang/Go简明教程/02-Go-Gin-简明教程.md)
- [Go2新特性简明教程](GoLang/Go简明教程/03-Go2新特性简明教程.md)
- [Protobuf简明教程](GoLang/Go简明教程/04-Go-Protobuf简明教程.md)
- [RPC&TLS鉴权简明教程](GoLang/Go简明教程/05-Go-RPC&TLS鉴权简明教程.md)
- [WebAssembly(Wasm)简明教程](GoLang/Go简明教程/06-Go-WebAssembly简明教程.md)
- [Test单元测试简明教程](GoLang/Go简明教程/07-Go-Test单元测试简明教程.md)
- [Mock(gomock)简明教程](GoLang/Go简明教程/08-Go-Mock简明教程.md)
- [Mmap-文件内存映射简明教程](GoLang/Go简明教程/09-Go-Mmap-文件内存映射简明教程.md)
- [Context并发编程简明教程](GoLang/Go简明教程/10-Go-Context并发编程简明教程.md)

### [GoLang基础](./GoLang/GoLang基础)

- [GoLang发展史](./GoLang/GoLang基础/01-GoLang发展史.md)
- [打印输出](./GoLang/GoLang基础/02-打印输出.md)
- [变量和常量](./GoLang/GoLang基础/03-变量和常量.md)
- [数据类型](./GoLang/GoLang基础/04-数据类型.md)
- [运算符](./GoLang/GoLang基础/05-运算符.md)
- [流程控制](./GoLang/GoLang基础/06-流程控制.md)
- [数组](./GoLang/GoLang基础/07-数组.md)
- [切片](./GoLang/GoLang基础/08-切片.md)
- [map](./GoLang/GoLang基础/09-map.md)
- [函数](./GoLang/GoLang基础/10-函数.md)
- [time包日期函数](./GoLang/GoLang基础/11-time包日期函数.md)
- [指针](./GoLang/GoLang基础/12-指针.md)
- [结构体](./GoLang/GoLang基础/13-结构体.md)
- [GoMod及包](./GoLang/GoLang基础/14-GoMod及包.md)
- [接口](./GoLang/GoLang基础/15-接口.md)
- [协程](./GoLang/GoLang基础/16-goroutine实现并行和并发.md)
- [反射](./GoLang/GoLang基础/17-反射.md)
- [文件和目录操作](./GoLang/GoLang基础/18-文件和目录操作.md)

### [7daysGoLang](GoLang/7daysGoLang)

- [目录](GoLang/7daysGoLang/README.md)
- 7天用Go从零实现Web框架 - Gee
    - 第一天：[前置知识(http.Handler接口)](./GoLang/7daysGoLang/gee-web/doc/gee-day1.md)
    - 第二天：[上下文设计(Context)](./GoLang/7daysGoLang/gee-web/doc/gee-day2.md)
    - 第三天：[Trie树路由(Router)](./GoLang/7daysGoLang/gee-web/doc/gee-day3.md)
    - 第四天：[分组控制(Group)](./GoLang/7daysGoLang/gee-web/doc/gee-day4.md)
    - 第五天：[中间件(Middleware)](./GoLang/7daysGoLang/gee-web/doc/gee-day5.md)
    - 第六天：[HTML模板(Template)](./GoLang/7daysGoLang/gee-web/doc/gee-day6.md)
    - 第七天：[错误恢复(Panic Recover)](./GoLang/7daysGoLang/gee-web/doc/gee-day7.md)

## [C++](./C++)

### [C++基础](./C++/基础)

- [C++初始](./C++/基础/01-C++初识.md)
  | [数据类型](./C++/基础/02-数据类型.md)
  | [运算符](./C++/基础/03-运算符.md)
- [流程控制](./C++/基础/04-流程控制.md)
  | [数组](./C++/基础/05-数组.md)
  | [函数](./C++/基础/06-函数.md)
- [指针](./C++/基础/07-指针.md)
  | [结构体](./C++/基础/08-结构体.md)
  | [内存分区模型](./C++/基础/09-内存分区模型.md)
- [引用](./C++/基础/10-引用.md)
  | [函数进阶](./C++/基础/11-函数进阶.md)
  | [类和对象](./C++/基础/12-类和对象.md)
  | [文件操作](./C++/基础/13-文件操作.md)

### [C++进阶](./C++/进阶)

- [模板](./C++/进阶/01-模板.md)
  | [STL初识](./C++/进阶/02-STL初识.md)
  | [STL函数对象](./C++/进阶/04-STL函数对象.md)

    - STL常用容器  
      [string](./C++/进阶/03-1-STL常用容器-string.md)
      | [vector](./C++/进阶/03-2-STL常用容器-vector.md)
      | [deque](./C++/进阶/03-3-STL常用容器-deque.md)
      | [stack](./C++/进阶/03-4-STL常用容器-stack.md)
      | [queue](./C++/进阶/03-5-STL常用容器-queue.md)
      | [list](./C++/进阶/03-6-STL常用容器-list.md)
      | [set-multiset](./C++/进阶/03-7-STL常用容器-set-multiset.md)
      | [map-multimap](./C++/进阶/03-8-STL常用容器-map-multimap.md)
    - STL常用算法  
      [遍历](./C++/进阶/05-1-STL常用算法-遍历.md)
      | [查找](./C++/进阶/05-2-STL常用算法-查找.md)
      | [排序](./C++/进阶/05-3-STL常用算法-排序.md)
      | [拷贝替换](./C++/进阶/05-4-STL常用算法-拷贝替换.md)
      | [算术生成](./C++/进阶/05-5-STL常用算法-算术生成.md)
      | [集合算法](./C++/进阶/05-6-STL常用算法-集合算法.md)

### [Linux环境编程](./C++/Linux环境编程)

- [gdb调试](./C++/Linux环境编程/01-gdb调试.md)
  | [make和Makefile](./C++/Linux环境编程/02-make和Makefile.md)
  | [CMakeLists入门](./C++/Linux环境编程/03-CMakeLists入门.md)
  | [CMakeLists进阶find_package的用法](./C++/Linux环境编程/05-CMakeLists进阶find_package的用法.md)
  | [静态库和动态库的制作和使用](./C++/Linux环境编程/04-静态库和动态库的制作和使用.md)

### [网络通信socket](./C++/网络通信socket)

- [socket概述](./C++/网络通信socket/01-socket概述.md)
  | [数据类型和相关库函数](./C++/网络通信socket/02-数据类型和相关库函数.md)
  | [网络字节序与主机字节序](./C++/网络通信socket/03-网络字节序与主机字节序.md)
  | [程序封装成类](./C++/网络通信socket/04-程序封装成类.md)
  | [多进程网络服务端](./C++/网络通信socket/05-多进程网络服务端.md)
- [TCP长连接和短连接](./C++/网络通信socket/06-TCP长连接和短连接.md)
  | [多线程网络服务端](./C++/网络通信socket/07-多线程网络服务端.md)
  | [性能测试](./C++/网络通信socket/08-性能测试.md)
  | [IO复用-select](./C++/网络通信socket/09-IO复用-select.md)
  | [IO复用-poll](./C++/网络通信socket/10-IO复用-poll.md)
  | [IO复用-epoll](./C++/网络通信socket/11-IO复用-epoll.md)

### [多进程](./C++/多进程)

- [进程概述](./C++/多进程/01-进程概述.md)
  | [孤儿进程和僵尸进程](./C++/多进程/02-孤儿进程和僵尸进程.md)
  | [守护进程](./C++/多进程/03-守护进程.md)
  | [进程间通信-管道](./C++/多进程/04-进程间通信-管道.md)
  | [进程间通信-信号](./C++/多进程/05-进程间通信-信号.md)
  | [进程间通信-共享内存](./C++/多进程/06-进程间通信-共享内存.md)
  | [进程间通信-信号量](./C++/多进程/07-进程间通信-信号量.md)

### [多线程](./C++/多线程)

- [多线程基础](./C++/多线程/01-多线程基础.md)
  | [线程同步](./C++/多线程/02-线程同步.md)
  | [多线程并发的网络服务](./C++/多线程/03-多线程并发的网络服务.md)
  | [线程同步案例](./C++/多线程/04-线程同步案例.md)

## [BehaviorTree](./BehaviorTree)

### [入门](./BehaviorTree/入门)

- [初始行为树](./BehaviorTree/入门/01-初始行为树.md)
- [行为树的基本知识点](./BehaviorTree/入门/02-行为树的基本知识点.md)：`xml`文件、`tick()`、节点种类：`ControlNode`、`DecoratorNode`、`ConditionNode`
  、`ActionNode`
- [基本类型Tree和TreeNode](./BehaviorTree/入门/03-库中基本类型Tree和TreeNode.md)
- [基本类型Factory和Blackboard](./BehaviorTree/入门/04-库中基本类型Factory和Blackboard.md)
    - [BehaviorTreeFactory](./BehaviorTree/入门/04-库中基本类型Factory和Blackboard.md#BehaviorTreeFactory)
      | [NodeBuilder](./BehaviorTree/入门/04-库中基本类型Factory和Blackboard.md#NodeBuilder)
      | [Blackboard](./BehaviorTree/入门/04-库中基本类型Factory和Blackboard.md#Blackboard)
      | [Port](./BehaviorTree/入门/04-库中基本类型Factory和Blackboard.md#Port)
- [DecoratorNodes源码解析](./BehaviorTree/入门/05-DecoratorNodes源码解析.md)
    - [DecoratorNode基类](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#DecoratorNode基类)
      | [BlackboardPreconditionNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#BlackboardPreconditionNode)
      | [DelayNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#DelayNode)
      | [ForceFailureNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#ForceFailureNode)
      | [InverterNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#InverterNode)
    - [KeepRunningUntilFailureNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#KeepRunningUntilFailureNode)
      | [RepeatNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#RepeatNode)
      | [RetryNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#RetryNode)
      | [SubtreeNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#SubtreeNode)
      | [SubtreePlusNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#SubtreePlusNode)
      | [TimeoutNode](./BehaviorTree/入门/05-DecoratorNodes源码解析.md#TimeoutNode)
- [ControlNodes源码解析](./BehaviorTree/入门/06-ControlNodes源码解析.md)
    - [ControlNode基类](./BehaviorTree/入门/06-ControlNodes源码解析.md#ControlNode基类)
      | [FallbackNode](./BehaviorTree/入门/06-ControlNodes源码解析.md#FallbackNode)
      | [ReactiveFallback](./BehaviorTree/入门/06-ControlNodes源码解析.md#ReactiveFallback)
      | [ParallelNode](./BehaviorTree/入门/06-ControlNodes源码解析.md#ParallelNode)
    - [IfThenElseNode](./BehaviorTree/入门/06-ControlNodes源码解析.md#IfThenElseNode)
      | [WhileDoElseNode](./BehaviorTree/入门/06-ControlNodes源码解析.md#WhileDoElseNode)
      | [SwitchNode](./BehaviorTree/入门/06-ControlNodes源码解析.md#SwitchNode)
      | [ManualSelectorNode](./BehaviorTree/入门/06-ControlNodes源码解析.md#ManualSelectorNode)
- [ControlNodes源码解析之Sequence](./BehaviorTree/入门/07-ControlNodes源码解析之Sequence.md)
    - [SequenceNode](./BehaviorTree/入门/07-ControlNodes源码解析之Sequence.md#SequenceNode)
      | [SequenceStarNode](./BehaviorTree/入门/07-ControlNodes源码解析之Sequence.md#SequenceStarNode)
      | [ReactiveSequence](./BehaviorTree/入门/07-ControlNodes源码解析之Sequence.md#ReactiveSequence)
- [ActionNode及同步和异步](./BehaviorTree/入门/08-ActionNode及同步和异步.md)
    - [ActionNodeBase](./BehaviorTree/入门/08-ActionNode及同步和异步.md#ActionNodeBase)
      | [SyncActionNode](./BehaviorTree/入门/08-ActionNode及同步和异步.md#SyncActionNode)
      | [SimpleActionNode](./BehaviorTree/入门/08-ActionNode及同步和异步.md#SimpleActionNode)
      | [AsyncActionNode](./BehaviorTree/入门/08-ActionNode及同步和异步.md#AsyncActionNode)
      | [StatefulActionNode](./BehaviorTree/入门/08-ActionNode及同步和异步.md#StatefulActionNode)
      | [CoroActionNode](./BehaviorTree/入门/08-ActionNode及同步和异步.md#CoroActionNode)
- [各种调试工具介绍](./BehaviorTree/入门/09-各种调试工具介绍.md)
    - [Groot](./BehaviorTree/入门/09-各种调试工具介绍.md#Groot)
      | [StdCoutLogger](./BehaviorTree/入门/09-各种调试工具介绍.md#StdCoutLogger)
      | [FileLogger](./BehaviorTree/入门/09-各种调试工具介绍.md#FileLogger)
      | [MinitraceLogger](./BehaviorTree/入门/09-各种调试工具介绍.md#MinitraceLogger)
      | [PublisherZMQ](./BehaviorTree/入门/09-各种调试工具介绍.md#PublisherZMQ)
      | [printTreeRecursively()内置函数](./BehaviorTree/入门/09-各种调试工具介绍.md#printTreeRecursively内置函数)
      | [debugMessage()内置函数](./BehaviorTree/入门/09-各种调试工具介绍.md#debugMessage内置函数)
- [Logger类实现原理解析-单例与观察者模式](./BehaviorTree/入门/10-Logger类实现原理解析（单例与观察者模式）.md)
- [行为树内外的数据传输](./BehaviorTree/入门/11-行为树内外的数据传输.md)
    - [树内即ports之间](./BehaviorTree/入门/11-行为树内外的数据传输.md#树内即ports之间)：`Blackboard`、`getInput`、`setOutput`、`SetBlackboard`
      | [subtree之间](./BehaviorTree/入门/11-行为树内外的数据传输.md#subtree之间)
      | [树与调用方之间](./BehaviorTree/入门/11-行为树内外的数据传输.md#树与调用方之间)
- [从xml创建加载行为树的过程分析](./BehaviorTree/入门/12-从xml创建加载行为树的过程分析.md)
- [自定义的用于枚举类型的SwitchNode](./BehaviorTree/入门/13-自定义的用于枚举类型的SwitchNode.md)
- [registerSimpleNode相关数据传输](./BehaviorTree/入门/14-registerSimpleNode相关数据传输.md)

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

- NoSQL概述 | Redis概述 | Mongo概述

#### Redeis

- [Redis安装与配置文件](Database/NoSQL/Redis/02-Redis安装与配置文件.md)
- [Redis基础](Database/NoSQL/Redis/01-Redis基础.md)
    - [Redis键](Database/NoSQL/Redis/01-Redis基础.md#redis键key)
      | [Redis字符串](Database/NoSQL/Redis/01-Redis基础.md#redis-字符串string)
      | [Redis哈希](Database/NoSQL/Redis/01-Redis基础.md#redis-哈希hash)
      | [Redis列表](Database/NoSQL/Redis/01-Redis基础.md#redis-列表list)
      | [Redis集合](Database/NoSQL/Redis/01-Redis基础.md#redis-集合set)
      | [Redis有序集合](Database/NoSQL/Redis/01-Redis基础.md#redis-有序集合sorted-set)
      | [Redis发布订阅](Database/NoSQL/Redis/01-Redis基础.md#redis-发布订阅)
    - [Redis多数据库](Database/NoSQL/Redis/01-Redis基础.md#redis多数据库)
      | [Redis事务](Database/NoSQL/Redis/01-Redis基础.md#redis-事务)
      | [Redis数据淘汰策略](Database/NoSQL/Redis/01-Redis基础.md#redis数据淘汰策略redisconf)
      | [Redis持久化](Database/NoSQL/Redis/01-Redis基础.md#redis持久化)
      | [Redis缓存与数据库Mysql一致性方案](Database/NoSQL/Redis/01-Redis基础.md#redis缓存与数据库mysql一致性方案)
- [Redis开发规范](Database/NoSQL/Redis/03-Redis开发规范.md)

## [机器学习](./MachineLearning/)

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
  | [pandas连接合并追加](./MachineLearning/MatplotlibNumpyPandas/Pandas_merge_concat_append.md)
  | [matplotlib基础](./MachineLearning/MatplotlibNumpyPandas/matplotlib基础.md)

### [深度学习](./MachineLearning/DeepLearning)
  - [安装](./MachineLearning/DeepLearning/chapter_installation/index.md)
  - [符号](./MachineLearning/DeepLearning/chapter_notation/index.md)
  - [引言](./MachineLearning/DeepLearning/chapter_introduction/index.md)
  - 预备知识
      - [数据操作](./MachineLearning/DeepLearning/chapter_preliminaries/ndarray.md)
      - [数据预处理](./MachineLearning/DeepLearning/chapter_preliminaries/pandas.md)
      - [线性代数](./MachineLearning/DeepLearning/chapter_preliminaries/linear-algebra.md)
      - [微积分](./MachineLearning/DeepLearning/chapter_preliminaries/calculus.md)
      - [自动微分](./MachineLearning/DeepLearning/chapter_preliminaries/autograd.md)
      - [概率](./MachineLearning/DeepLearning/chapter_preliminaries/probability.md)
      - [查阅文档](./MachineLearning/DeepLearning/chapter_preliminaries/lookup-api.md)
  - 线性神经网络
      - [线性回归](./MachineLearning/DeepLearning/chapter_linear-networks/linear-regression.md)
      - [线性回归的从零开始实现](./MachineLearning/DeepLearning/chapter_linear-networks/linear-regression-scratch.md)
      - [线性回归的简洁实现](./MachineLearning/DeepLearning/chapter_linear-networks/linear-regression-concise.md)
      - [softmax回归](./MachineLearning/DeepLearning/chapter_linear-networks/softmax-regression.md)
      - [图像分类数据集](./MachineLearning/DeepLearning/chapter_linear-networks/image-classification-dataset.md)
      - [softmax回归的从零开始实现](./MachineLearning/DeepLearning/chapter_linear-networks/softmax-regression-scratch.md)
  - 多层感知机
      - [多层感知机](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/mlp.md)
      - [多层感知机的从零开始实现](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/mlp-scratch.md)
      - [多层感知机的简洁实现](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/mlp-concise.md)
      - [模型选择、欠拟合和过拟合](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/underfit-overfit.md)
      - [权重衰减](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/weight-decay.md)
      - [暂退法（Dropout）](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/dropout.md)
      - [前向传播、反向传播和计算图](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/backprop.md)
      - [数值稳定性和模型初始化](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/numerical-stability-and-init.md)
      - [环境和分布偏移](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/environment.md)
      - [实战Kaggle比赛：预测房价](./MachineLearning/DeepLearning/chapter_multilayer-perceptrons/kaggle-house-price.md)
  - 深度学习计算
      - [层和块](./MachineLearning/DeepLearning/chapter_deep-learning-computation/model-construction.md)
      - [参数管理](./MachineLearning/DeepLearning/chapter_deep-learning-computation/parameters.md)
      - [延后初始化](./MachineLearning/DeepLearning/chapter_deep-learning-computation/deferred-init.md)
      - [自定义层](./MachineLearning/DeepLearning/chapter_deep-learning-computation/custom-layer.md)
      - [读写文件](./MachineLearning/DeepLearning/chapter_deep-learning-computation/read-write.md)
      - [GPU](./MachineLearning/DeepLearning/chapter_deep-learning-computation/use-gpu.md)
  - 卷积神经网络
      - [从全连接层到卷积](./MachineLearning/DeepLearning/chapter_convolutional-neural-networks/why-conv.md)
      - [图像卷积](./MachineLearning/DeepLearning/chapter_convolutional-neural-networks/conv-layer.md)
      - [填充和步幅](./MachineLearning/DeepLearning/chapter_convolutional-neural-networks/padding-and-strides.md)
      - [多输入多输出通道](./MachineLearning/DeepLearning/chapter_convolutional-neural-networks/channels.md)
      - [汇聚层](./MachineLearning/DeepLearning/chapter_convolutional-neural-networks/pooling.md)
      - [卷积神经网络（LeNet）](./MachineLearning/DeepLearning/chapter_convolutional-neural-networks/lenet.md)
  - 现代卷积神经网络
      - [深度卷积神经网络（AlexNet）](./MachineLearning/DeepLearning/chapter_convolutional-modern/alexnet.md)
      - [使用块的网络（VGG）](./MachineLearning/DeepLearning/chapter_convolutional-modern/vgg.md)
      - [网络中的网络（NiN）](./MachineLearning/DeepLearning/chapter_convolutional-modern/nin.md)
      - [含并行连结的网络（GoogLeNet）](./MachineLearning/DeepLearning/chapter_convolutional-modern/googlenet.md)
      - [批量规范化](./MachineLearning/DeepLearning/chapter_convolutional-modern/batch-norm.md)
      - [残差网络（ResNet）](./MachineLearning/DeepLearning/chapter_convolutional-modern/resnet.md)
      - [稠密连接网络（DenseNet）](./MachineLearning/DeepLearning/chapter_convolutional-modern/densenet.md)
  - 循环神经网络
      - [序列模型](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/sequence.md)
      - [文本预处理](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/text-preprocessing.md)
      - [语言模型和数据集](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/language-models-and-dataset.md)
      - [循环神经网络](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/rnn.md)
      - [循环神经网络的从零开始实现](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/rnn-scratch.md)
      - [循环神经网络的简洁实现](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/rnn-concise.md)
      - [通过时间反向传播](./MachineLearning/DeepLearning/chapter_recurrent-neural-networks/bptt.md)
  - 现代循环神经网络
    - [门控循环单元（GRU）](./MachineLearning/DeepLearning/chapter_recurrent-modern/gru.md)
    - [长短期记忆网络（LSTM）](./MachineLearning/DeepLearning/chapter_recurrent-modern/lstm.md)
    - [深度循环神经网络](./MachineLearning/DeepLearning/chapter_recurrent-modern/deep-rnn.md)
    - [双向循环神经网络](./MachineLearning/DeepLearning/chapter_recurrent-modern/bi-rnn.md)
    - [机器翻译与数据集](./MachineLearning/DeepLearning/chapter_recurrent-modern/machine-translation-and-dataset.md)
    - [编码器-解码器架构](./MachineLearning/DeepLearning/chapter_recurrent-modern/encoder-decoder.md)
    - [序列到序列学习（seq2seq）](./MachineLearning/DeepLearning/chapter_recurrent-modern/seq2seq.md)
    - [束搜索](./MachineLearning/DeepLearning/chapter_recurrent-modern/beam-search.md)
  - 注意力机制
    - [注意力提示](./MachineLearning/DeepLearning/chapter_attention-mechanisms/attention-cues.md)
    - [注意力汇聚：Nadaraya-Watson 核回归](./MachineLearning/DeepLearning/chapter_attention-mechanisms/nadaraya-waston_origin.md)
    - [注意力评分函数](./MachineLearning/DeepLearning/chapter_attention-mechanisms/attention-scoring-functions.md)
    - [Bahdanau 注意力](./MachineLearning/DeepLearning/chapter_attention-mechanisms/bahdanau-attention.md)
    - [多头注意力](./MachineLearning/DeepLearning/chapter_attention-mechanisms/multihead-attention.md)
    - [自注意力和位置编码](./MachineLearning/DeepLearning/chapter_attention-mechanisms/self-attention-and-positional-encoding.md)
    - [Transformer](./MachineLearning/DeepLearning/chapter_attention-mechanisms/transformer.md)
  - 优化算法
    - [优化和深度学习](./MachineLearning/DeepLearning/chapter_optimization/optimization-intro.md)
    - [凸性](./MachineLearning/DeepLearning/chapter_optimization/convexity.md)
    - [梯度下降](./MachineLearning/DeepLearning/chapter_optimization/gd.md)
    - [随机梯度下降](./MachineLearning/DeepLearning/chapter_optimization/sgd.md)
    - [小批量随机梯度下降](./MachineLearning/DeepLearning/chapter_optimization/minibatch-sgd.md)
    - [动量法](./MachineLearning/DeepLearning/chapter_optimization/momentum.md)
    - [AdaGrad算法](./MachineLearning/DeepLearning/chapter_optimization/adagrad.md)
    - [RMSProp算法](./MachineLearning/DeepLearning/chapter_optimization/rmsprop.md)
    - [Adadelta](./MachineLearning/DeepLearning/chapter_optimization/adadelta.md)
    - [Adam算法](./MachineLearning/DeepLearning/chapter_optimization/adam.md)
    - [学习率调度器](./MachineLearning/DeepLearning/chapter_optimization/lr-scheduler.md)
  - 计算性能
    - [编译器和解释器](./MachineLearning/DeepLearning/chapter_computational-performance/hybridize.md)
    - [异步计算](./MachineLearning/DeepLearning/chapter_computational-performance/async-computation.md)
    - [自动并行](./MachineLearning/DeepLearning/chapter_computational-performance/auto-parallelism.md)
    - [硬件](./MachineLearning/DeepLearning/chapter_computational-performance/hardware.md)
    - [多GPU训练](./MachineLearning/DeepLearning/chapter_computational-performance/multiple-gpus.md)
    - [多GPU的简洁实现](./MachineLearning/DeepLearning/chapter_computational-performance/multiple-gpus-concise.md)
    - [参数服务器](./MachineLearning/DeepLearning/chapter_computational-performance/parameterserver.md)
  - 计算机视觉
    - [图像增广](./MachineLearning/DeepLearning/chapter_computer-vision/image-augmentation.md)
    - [微调](./MachineLearning/DeepLearning/chapter_computer-vision/fine-tuning.md)
    - [目标检测和边界框](./MachineLearning/DeepLearning/chapter_computer-vision/bounding-box.md)
    - [锚框](./MachineLearning/DeepLearning/chapter_computer-vision/anchor.md)
    - [多尺度目标检测](./MachineLearning/DeepLearning/chapter_computer-vision/multiscale-object-detection.md)
    - [目标检测数据集](./MachineLearning/DeepLearning/chapter_computer-vision/object-detection-dataset.md)
    - [单发多框检测（SSD）](./MachineLearning/DeepLearning/chapter_computer-vision/ssd.md)
    - [区域卷积神经网络（R-CNN）系列](./MachineLearning/DeepLearning/chapter_computer-vision/rcnn.md)
    - [语义分割和数据集](./MachineLearning/DeepLearning/chapter_computer-vision/semantic-segmentation-and-dataset.md)
    - [转置卷积](./MachineLearning/DeepLearning/chapter_computer-vision/transposed-conv.md)
    - [全卷积网络](./MachineLearning/DeepLearning/chapter_computer-vision/fcn.md)

## [Web前端](./Web/)

- [Web前端入门](./Web/Web前端概述.md)
  | [HTML5](./Web/HTML5)
  | [CSS](./Web/CSS)
  | [JavaScript](./Web/JavaScript)
- [jQuery](./Web/JavaScript/框架)
  | [jQuery-UI](./Web/JavaScript/框架/jQuery-UI.md)
  | [Vue.js](./Web/JavaScript/框架)
  | [RESTful规范](./Web/RESTful.md)

## ROS

### [ROS1](ROS)

#### [基础](ROS/基础)

- [Windows安装rospy](ROS/基础/00-Windows安装rospy.md)
- [什么是ROS](ROS/基础/01-什么是ROS.md)
- [Catkin工作空间编译系统](ROS/基础/02-Catkin工作空间与编译系统.md)
- [ROS通信架构上](ROS/基础/03-ROS通信架构上.md)
- [ROS通信架构下](ROS/基础/04-ROS通信架构下.md)
- [常用工具](ROS/基础/05-常用工具.md)
- [roscpp](ROS/基础/06-roscpp.md)
- [rospy](ROS/基础/07-rospy.md)
- [静态和动态参数使用](ROS/基础/08-静态和动态参数使用.md)

#### [进阶](ROS/进阶)

##### [源码分析](ROS/进阶/源码分析)

- [ros-logs](ROS/进阶/源码分析/00-ros-logs.md)
- [roscore与Master启动](ROS/进阶/源码分析/01-roscore与Master启动.md)
- [roslaunch](ROS/进阶/源码分析/02-roslaunch.md)
- [process_monitoring](ROS/进阶/源码分析/03-process_monitoring.md)
- [topic](ROS/进阶/源码分析/04-topic.md)
- [service](ROS/进阶/源码分析/05-service.md)

### [ROS2](ROS2)

#### ROS2入门篇

##### ROS2介绍与安装

- 基础
    - [ROS2介绍与安装](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/基础/001-ROS2介绍与安装.md)
    - [在虚拟机中安装Ubuntu](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/基础/002-在虚拟机中安装Ubuntu.md)
    - [玩转Ubuntu之常用指令](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/基础/003-玩转Ubuntu之常用指令.md)
    - [玩转Ubuntu之编程工具](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/基础/004-玩转Ubuntu之编程工具.md)
    - [玩转Ubuntu之常用软件](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/基础/005-玩转Ubuntu之常用软件.md)
- 入门
    - [ROS2的前世今生](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/入门/001-ROS2的前世今生.md)
    - [安装ROS2-初体验](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/入门/002-安装ROS2-初体验.md)
- 进阶
    - [ROS2系统架构](ROS2/ROS2入门篇/第1章-ROS2介绍与安装/进阶/001-ROS2系统架构.md)

##### ROS2第一个节点

- 基础
    - [cplus](ROS2/ROS2入门篇/第2章-ROS2第一个节点/基础/001-cplus.md)
    - [python](ROS2/ROS2入门篇/第2章-ROS2第一个节点/基础/002-python.md)
- 入门
    - [节点介绍-工作空间-构建工具Colcon](ROS2/ROS2入门篇/第2章-ROS2第一个节点/入门/001-节点介绍-工作空间-构建工具Colcon.md)
    - [使用RCLCPP和RCLPY编写节点](ROS2/ROS2入门篇/第2章-ROS2第一个节点/入门/002-使用RCLCPP和RCLPY编写节点.md)
- 进阶
    - [面向对象编程思想](ROS2/ROS2入门篇/第2章-ROS2第一个节点/进阶/001-面向对象编程思想.md)
    - [Colcon使用进阶](ROS2/ROS2入门篇/第2章-ROS2第一个节点/进阶/002-Colcon使用进阶.md)
    - [ROS2节点发现与多机通信](ROS2/ROS2入门篇/第2章-ROS2第一个节点/进阶/003-ROS2节点发现与多机通信.md)

##### ROS2节点通信之话题与服务

- 基础
    - [从底层理解通信及通信中间件ZMQ](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/基础/001-从底层理解通信及通信中间件ZMQ.md)
- 入门
    - [topic](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/001-topic.md)
    - [topic-rclcpp](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/002-topic-rclcpp.md)
    - [topic-rclpy](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/003-topic-rclpy.md)
    - [service](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/004-service.md)
    - [service-rclcpp](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/005-service-rclcpp.md)
    - [service-rclpy](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/006-service-rclpy.md)
    - [interface](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/007-interface.md)
    - [interface-rclcpp](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/008-interface-rclcpp.md)
    - [interface-rclpy](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/入门/009-interface-rclpy.md)
- 进阶
    - [原始数据类型与包装类型](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/进阶/001-原始数据类型与包装类型.md)
    - [通信质量Qos配置指南](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/进阶/002-通信质量Qos配置指南.md)
    - [DDS进阶之Fast-DDS环境搭建](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/进阶/003-DDS进阶之Fast-DDS环境搭建.md)
    - [使用DDS进行订阅发布](ROS2/ROS2入门篇/第3章-ROS2节点通信之话题与服务/进阶/004-使用DDS进行订阅发布.md)

##### ROS2节点通信之参数与动作

- 基础
    - [开环控制与闭环控制](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/基础/001-开环控制与闭环控制.md)
- 入门
    - [param](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/001-param.md)
    - [param-rclcpp](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/002-param-rclcpp.md)
    - [param-rclpy](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/003-param-rclpy.md)
    - [action](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/004-action.md)
    - [action-rclcpp](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/005-action-rclcpp.md)
    - [action-rclpy](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/006-action-rclpy.md)
    - [通信机制对比总结](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/入门/007-通信机制对比总结.md)
- 进阶
    - [ROS参数通信原理介绍](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/进阶/001-ROS参数通信原理介绍.md)
    - [生命周期节点](ROS2/ROS2入门篇/第4章-ROS2通信之参数与动作/进阶/002-生命周期节点.md)

##### ROS2常用工具

- [启动管理工具-Launch](ROS2/ROS2入门篇/第5章-ROS2常用工具/001-启动管理工具-Launch.md)
- [ROS2命令行工具](ROS2/ROS2入门篇/第5章-ROS2常用工具/002-ROS2命令行工具.md)
- [RVIZ2](ROS2/ROS2入门篇/第5章-ROS2常用工具/003-RVIZ2.md)
- [RQT工具](ROS2/ROS2入门篇/第5章-ROS2常用工具/004-RQT工具.md)
- [时光记录仪之rosbag2](ROS2/ROS2入门篇/第5章-ROS2常用工具/005-时光记录仪之rosbag2.md)
- [兼容仿真工具-Gazebo](ROS2/ROS2入门篇/第5章-ROS2常用工具/006-兼容仿真工具-Gazebo.md)

#### 机器人学篇

##### 运动学基础

- 基础-数学基础
    - [矩阵与矩阵运算](ROS2/机器人学篇/第6章-运动学基础/基础-数学基础/001-矩阵与矩阵运算.md)
    - [MiniConda与Jupyter介绍安装](ROS2/机器人学篇/第6章-运动学基础/基础-数学基础/002-MiniConda与Jupyter介绍安装.md)
    - [矩阵运算实战](ROS2/机器人学篇/第6章-运动学基础/基础-数学基础/003-矩阵运算实战.md)
- 入门-机器人运动学
    - [空间坐标描述](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/001-空间坐标描述.md)
    - [空间坐标描述实战](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/002-空间坐标描述实战.md)
    - [姿态的不同表示](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/003-姿态的不同表示.md)
    - [姿态转换实战](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/004-姿态转换实战.md)
    - [齐次坐标变换](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/005-齐次坐标变换.md)
    - [齐次坐标变换实战](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/006-齐次坐标变换实战.md)
    - [机器人运动学介绍](ROS2/机器人学篇/第6章-运动学基础/入门-机器人运动学/007-机器人运动学介绍.md)

##### ROS2运动学

- [TF2介绍及RVIZ-TF组件](ROS2/机器人学篇/第7章-ROS2运动学/001-TF2介绍及RVIZ-TF组件.md)
- [坐标变换发布监听Python和Cpp实现](ROS2/机器人学篇/第7章-ROS2运动学/002-坐标变换发布监听Python和Cpp实现.md)

#### 建模仿真篇

##### 机器人建模

- [URDF统一机器人建模语言](ROS2/建模仿真篇/第8章-机器人建模/001-URDF统一机器人建模语言.md)
- [RVIZ2可视化移动机器人模型](ROS2/建模仿真篇/第8章-机器人建模/002-RVIZ2可视化移动机器人模型.md)
- [创建一个两轮差速模型](ROS2/建模仿真篇/第8章-机器人建模/003-创建一个两轮差速模型.md)
- [通过JointStates控制RVIZ2关节](ROS2/建模仿真篇/第8章-机器人建模/004-通过JointStates控制RVIZ2关节.md)

##### 机器人仿真

- 入门
    - [机器人仿真介绍](ROS2/建模仿真篇/第9章-机器人仿真/入门/001-机器人仿真介绍.md)
    - [为机器人URDF模型注入物理属性](ROS2/建模仿真篇/第9章-机器人仿真/入门/002-为机器人URDF模型注入物理属性.md)
    - [常见几何物体URDF仿真惯性参数怎么配](ROS2/建模仿真篇/第9章-机器人仿真/入门/003-常见几何物体URDF仿真惯性参数怎么配.md)
    - [在Gazebo加载机器人模型](ROS2/建模仿真篇/第9章-机器人仿真/入门/004-在Gazebo加载机器人模型.md)
    - [Gazebo仿真插件之两轮差速](ROS2/建模仿真篇/第9章-机器人仿真/入门/005-Gazebo仿真插件之两轮差速.md)
    - [Gazebo仿真插件之IMU](ROS2/建模仿真篇/第9章-机器人仿真/入门/006-Gazebo仿真插件之IMU.md)
    - [Gazebo仿真插件之超声波](ROS2/建模仿真篇/第9章-机器人仿真/入门/007-Gazebo仿真插件之激光雷达.md)
    - [Gazebo仿真插件之超声波](ROS2/建模仿真篇/第9章-机器人仿真/入门/008-Gazebo仿真插件之超声波.md)
    - [Gazebo仿真环境搭建](ROS2/建模仿真篇/第9章-机器人仿真/入门/009-Gazebo仿真环境搭建.md)
- 进阶
    - [Gazebo仿真进阶教程之传感器高斯噪声](ROS2/建模仿真篇/第9章-机器人仿真/进阶/001-Gazebo仿真进阶教程之传感器高斯噪声.md)

#### Nav2导航篇

##### SLAM建图

- 基础
    - [图像常见格式及存储](ROS2/Nav2导航篇/第10章-SLAM建图/基础/001-图像常见格式及存储.md)
- 入门
    - [SLAM前世今生](ROS2/Nav2导航篇/第10章-SLAM建图/入门/001-SLAM前世今生.md)
    - [Carto介绍及安装](ROS2/Nav2导航篇/第10章-SLAM建图/入门/002-Carto介绍及安装.md)
    - [配置FishBot进行建图](ROS2/Nav2导航篇/第10章-SLAM建图/入门/003-配置FishBot进行建图.md)
- 进阶
    - [ROS2地图加载与编辑](ROS2/Nav2导航篇/第10章-SLAM建图/进阶/001-ROS2地图加载与编辑.md)
    - [使用纯雷达定位建图](ROS2/Nav2导航篇/第10章-SLAM建图/进阶/002-使用纯雷达定位建图.md)

##### Nav2导航仿真实战

- 基础
- 入门
    - [Nav2导航框架介绍](ROS2/Nav2导航篇/第11章-Nav2导航仿真实战/入门/001-Nav2导航框架介绍.md)
    - [为FishBot配置Nav2](ROS2/Nav2导航篇/第11章-Nav2导航仿真实战/入门/002-为FishBot配置Nav2.md)
    - [使用FishBot进行自主导航](ROS2/Nav2导航篇/第11章-Nav2导航仿真实战/入门/003-使用FishBot进行自主导航.md)
    - [使用Nav2导航API进行导航](ROS2/Nav2导航篇/第11章-Nav2导航仿真实战/入门/004-使用Nav2导航API进行导航.md)
- 进阶
    - [使用EKF融合里程计和IMU](ROS2/Nav2导航篇/第11章-Nav2导航仿真实战/进阶/001-使用EKF融合里程计和IMU.md)

##### Nav2进阶实践

#### ROS2硬件控制篇

##### 嵌入式开发之从点灯开始

- 基础
    - [嵌入式开发介绍与环境搭建](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/基础/001-嵌入式开发介绍与环境搭建.md)
    - [HelloWord-串口通信-接收实验](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/基础/002-第一个HelloWord工程-串口通信-接收实验.md)
- 入门
    - [看懂LED驱动电路-GPIO-ADC](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/入门/001-看懂LED驱动电路-GPIO控制-学会使用ADC.md)
- 进阶
    - [学会安装第三方开源库](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/进阶/001-学会安装第三方开源库.md)
    - [使用开源库驱动IMU](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/进阶/002-使用开源库驱动IMU.md)
    - [面向对象封装IMU驱动](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/进阶/003-学会面向对象编程-封装IMU驱动.md)
    - [使用开源库驱动OLED](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/进阶/004-使用开源库驱动OLED.md)
    - [I2C通信-点亮OLED](ROS2/ROS2硬件控制篇/第13章-嵌入式开发之从点灯开始/进阶/005-I2C通信-点亮OLED.md)

##### 接入ROS2-MicroROS

- 基础
    - [MicroROS介绍与服务安装](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/基础/001-MicroROS介绍与服务安装.md)
    - [第一个MicroROS节点](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/基础/002-第一个MicroROS节点.md)
- 入门
    - [话题订阅-控制LED](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/入门/001-话题订阅-控制LED.md)
    - [MicroROS话题发布实现](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/入门/002-MicroROS话题发布实现.md)
    - [MicroROS服务通信服务端实现](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/入门/003-MicroROS服务通信服务端实现.md)
- 进阶
    - [控制OLED自定义消息接口](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/进阶/001-控制OLED自定义消息接口.md)
    - [做个时钟-系统时间同步](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/进阶/002-做个时钟-系统时间同步.md)
    - [无线通讯-了解传输原理](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/进阶/003-无线通讯-了解传输原理.md)
    - [使用双核运行MicroROS](ROS2/ROS2硬件控制篇/第14章-接入ROS2-MicroROS/进阶/004-榨干性能-使用双核运行MicroROS.md)

##### ROS2硬件实战-自制简易雷达

- [测量距离学会超声波传感器](ROS2/ROS2硬件控制篇/第15章-ROS2硬件实战-自制简易雷达/001-测量距离学会超声波传感器.md)
- [控制舵机学会使用执行器](ROS2/ROS2硬件控制篇/第15章-ROS2硬件实战-自制简易雷达/002-控制舵机学会使用执行器.md)
- [舵机和超声波循环扫描](ROS2/ROS2硬件控制篇/第15章-ROS2硬件实战-自制简易雷达/003-舵机和超声波循环扫描.md)
- [可视化点云雷达消息合成](ROS2/ROS2硬件控制篇/第15章-ROS2硬件实战-自制简易雷达/004-可视化点云雷达消息合成.md)

#### 两轮差速移动机器人开发篇

##### 移动机器人控制系统搭建

- [移动机器人底盘结构介绍](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/001-移动机器人底盘结构介绍.md)
- [从H桥说起-电机驱动原理](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/002-从H桥说起-电机驱动原理介绍.md)
- [电机控制之正反转实验](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/003-电机控制之正反转实验.md)
- [电机控制之速度控制实验](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/004-电机控制之速度控制实验.md)
- [电机控制之使用开源库驱动多路电机](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/005-电机控制之使用开源库驱动多路电机.md)
- [做个遥控车订阅ROS2Twist](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/006-做个遥控车订阅ROS2Twist.md)
- [速度测量-编码器-脉冲测量与校准](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/007-速度测量-编码器-脉冲测量与校准.md)
- [速度转换-机器人最大速度测量](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/008-速度转换-机器人最大速度测量.md)
- [控制速度-PID控制器实现](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/009-控制速度-PID控制器实现.md)
- [两轮差速运动学正逆解](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/010-两轮差速运动学正逆解.md)
- [里程计计算-速度积分](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/011-里程计计算-速度积分.md)
- [采用MicroROS发布里程计](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/012-采用MicroROS发布里程计.md)
- [项目总结与扩展-源码编译Agent](ROS2/两轮差速移动机器人开发篇/第16章-移动机器人控制系统搭建/013-项目总结与扩展-源码编译Agent.md)

##### 建图与导航

- slam
    - [学会驱动雷达](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/slam/001-可视化雷达点云-学会驱动雷达.md)
    - [了解ROS标准REP105](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/slam/002-建图前准备1-了解ROS标准REP105.md)
    - [发布odom的TF](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/slam/003-建图前准备2-发布odom的TF.md)
    - [准备URDF](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/slam/004-建图前准备3-准备URDF.md)
    - [使用SLAM_TOOLBOX完成建图](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/slam/005-使用SLAM_TOOLBOX完成建图.md)
- navigation
    - [Nav2安装与配置](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/navigation/001-Nav2安装与配置.md)
    - [启动导航和单点与路点导航](ROS2/两轮差速移动机器人开发篇/第17章-建图与导航实现/navigation/002-启动导航和单点与路点导航.md)

##### ROS2常用代码模板

- [rclcpp](ROS2/ROS2常用代码模板/001-rclcpp.md)
- [rclpy](ROS2/ROS2常用代码模板/002-rclpy.md)
- [urdf](ROS2/ROS2常用代码模板/003-urdf.md)

## Navigation2

### 源码分析

- [核心定义](Navigation2/源码分析/001-核心定义.md)
- [通用插件系统管理](Navigation2/源码分析/002-通用插件系统管理.md)
- [行为树及节点](Navigation2/源码分析/003-行为树及节点.md)
- [地图和定位](Navigation2/源码分析/004-地图和定位.md)
- [规划器服务](Navigation2/源码分析/005-规划器服务.md)
- [平滑器服务](Navigation2/源码分析/006-平滑器服务.md)
- [控制器服务](Navigation2/源码分析/007-控制器服务.md)
- [应用及工具](Navigation2/源码分析/008-应用及工具.md)

## 通信工具

### LCM
  - [lcm基础](CommunicationTools/lcm/lcm基础.md)


## [工具](./Utils/)

### [Docker](./Utils/docker/Docker.md)

- [基础](./Utils/docker/Docker.md)
- [docker-history](./Utils/docker/01-docker-history.md)
- [container-diff](./Utils/docker/02-container-diff.md)
- [制作容器镜像的最佳实践](./Utils/docker/03-制作容器镜像的最佳实践.md)
- [制作Python_Docker镜像的最佳实践](./Utils/docker/04-制作Python_Docker镜像的最佳实践.md)
- [Docker入门PDF](./Utils/docker/Docker入门教程.pdf)
- [Docker部署Django Uwsgi+Nginx+MySQL+Redis](./Utils/docker/django_demo_docker/README.md)
- [Docker简单部署Django的FastDFS](./Utils/docker/FastDFS.md)

### [Git](./Utils/Git基本命令.md)

- 帮助信息 / git配置(全局配置) / 初始化项目 / 查看状态 / 添加文件 / 提交文件 / 查看提交日志
- 查看文件修改前后的区别 / git跟踪rename文件和移动文件 / 删除文件 / 恢复文件 / 恢复提交
- 重置提交指针 / 查看、创建、切换分支 / 查看两个分支的区别 / 合并分支 / 解决合并冲突 / 重命名和删除分支
- 保存修改进度 / 添加别名 / 全局忽略跟踪文件 / 项目级忽略文件 / 创建远程版本库 / 推送版本库
- 修改远程仓库地址 / 克隆版本库到本地 / 更新本地版本库 / 基于版本库开发自己的版本库 / 添加pull request / 添加贡献者

### [Nginx](Utils/Nginx/Nginx基础.md)

- [基础](Utils/Nginx/Nginx基础.md)
- [进阶](Utils/Nginx/Nginx进阶.md)

### [IDE](./Utils//IDE/)

- [VSCode](./Utils/IDE/vscode.md)

### [工作所学技能或知识](Works)

- [福建路阳有限公司](Works/01-贞仕.md)
- [上海快仓自动化有限公司](Works/02-快仓.md)

## [面试](InterviewPreparation/README.md)

- [技术面试必备基础知识](InterviewPreparation/TechnicalInterviews/TechnicalInterviews.md)
    - [操作系统](InterviewPreparation/TechnicalInterviews/01-操作系统.md)
    - [网络](InterviewPreparation/TechnicalInterviews/02-网络.md)
    - [数据库](InterviewPreparation/TechnicalInterviews/03-数据库.md)
    - [系统设计](InterviewPreparation/TechnicalInterviews/04-系统设计.md)
- [数据结构与算法](InterviewPreparation/DataStructuresAlgorithms/DataStructuresAlgorithms.md)
    - 算法
        - 基础
            - [算法](InterviewPreparation/DataStructuresAlgorithms/01-算法.md)
            - [选择、冒泡和插入排序](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/01-选择、冒泡和插入排序.md)
            - [归并、快速排序](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/02-归并、快速排序.md)
            - [堆和桶排序以及排序总结](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/03-堆、桶排序以及排序总结.md)
            - [链表](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/04-链表.md)
            - [二叉树](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/05-二叉树.md)
            - [图](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/06-图.md)
            - [前缀树和贪心算法](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/07-前缀树和贪心算法.md)
            - [暴力递归](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础/08-暴力递归.md)
            - [算法图解](InterviewPreparation/DataStructuresAlgorithms/Algorithm/算法图解.pdf)
        - 基础提升
            - [哈希函数和哈希表](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/01-哈希函数和哈希表.md)
            - [字符串匹配算法](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/02-字符串匹配算法.md)
            - [滑动窗口-单调栈](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/03-滑动窗口-单调栈-非常重要.md)
            - [Morris遍历](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/04-Morris遍历.md)
            - [大数据题目-资源限制类](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/05-大数据题目-资源限制类.md)
            - [位运算](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/06-位运算.md)
            - [暴力递归到动态规划](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/07-暴力递归到动态规划.md)
            - [有序表](InterviewPreparation/DataStructuresAlgorithms/Algorithm/基础提升/08-有序表.md)
        - 数据滤波
            - [卡尔曼滤波](InterviewPreparation/DataStructuresAlgorithms/Algorithm/Wavefiltering/KalmanFiltering.md)
    - 数据结构
        - [线性表](InterviewPreparation/DataStructuresAlgorithms/02-线性表.md)
        - [栈](InterviewPreparation/DataStructuresAlgorithms/03-栈.md)
        - [队列](InterviewPreparation/DataStructuresAlgorithms/04-队列.md)
        - [特殊矩阵压缩存储](InterviewPreparation/DataStructuresAlgorithms/05-特殊矩阵压缩存储.md)
        - [串](InterviewPreparation/DataStructuresAlgorithms/06-串.md)
        - [树与二叉树](InterviewPreparation/DataStructuresAlgorithms/07-树与二叉树.md)
- 面试题
    - [2020](InterviewPreparation/ClassicQuestion/20201027/README.md)

## [理财](FinancialManagement)

- 基金
    - 基金基础
        - [纯债基金](FinancialManagement/基金/基金基础-货基债基股基/01-纯债基金.md)
        - [股票型基金](FinancialManagement/基金/基金基础-货基债基股基/02-股票型基金.md)
        - [大数据指数基金](FinancialManagement/基金/基金基础-货基债基股基/03-大数据指数基金.md)
        - [ETF基金](FinancialManagement/基金/基金基础-货基债基股基/04-ETF基金.md)
        - [LOF基金](FinancialManagement/基金/基金基础-货基债基股基/05-LOF基金.md)
        - [四大行业指数](FinancialManagement/基金/基金基础-货基债基股基/06-四大行业指数.md)
    - 基金进价
        - [分级基金](FinancialManagement/基金/基金进价-特种基金：分级基金QDII基金量化基金等/01-分级基金.md)
        - [避险基金](FinancialManagement/基金/基金进价-特种基金：分级基金QDII基金量化基金等/02-避险基金.md)
        - [量化基金](FinancialManagement/基金/基金进价-特种基金：分级基金QDII基金量化基金等/03-量化基金.md)
        - [QDII基金](FinancialManagement/基金/基金进价-特种基金：分级基金QDII基金量化基金等/04-QDII基金.md)
        - [FOF基金](FinancialManagement/基金/基金进价-特种基金：分级基金QDII基金量化基金等/05-FOF基金.md)
    - 其他
        - [基金定投](FinancialManagement/基金/基金定投/01-基金定投.md)
        - [基金投资术语](FinancialManagement/基金/00-投资术语.md)
        - [投资误区](FinancialManagement/基金/00-投资误区.md)
        - [全球配置](FinancialManagement/基金/基金全球配置大法/01-全球配置.md)
        - [场外基金开户和买卖实操](FinancialManagement/基金/基金全球配置大法/02-场外基金开户和买卖实操.md)
        - [场内基金开户和买卖实操](FinancialManagement/基金/基金全球配置大法/03-场内基金开户和买卖实操.md)
        - [看懂股票行情](FinancialManagement/基金/基金全球配置大法/04-看懂股票行情.md)
        - [基金套牢怎么办](FinancialManagement/基金/基金全球配置大法/05-基金套牢怎么办.md)
- 股票
    - [股票市场常用名称解释](FinancialManagement/股票/01-股票市场常用名称解释.md)
    - 看盘
        - [早盘](FinancialManagement/股票/02-看盘-早盘.md)
        - [盘后](FinancialManagement/股票/03-看盘-盘后.md)
        - [大盘分时图分析技巧](FinancialManagement/股票/04-看盘-大盘分时图分析技巧.md)
        - [K线图](FinancialManagement/股票/05-看盘-K线图.md)
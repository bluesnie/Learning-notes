###### datetime:2019/6/14 9:04
###### author:nzb

# 学习笔记

## Python应用领域和就业形势分析

简单的说，Python是一个“优雅”、“明确”、“简单”的编程语言。

 - 学习曲线低，非专业人士也能上手
 - 开源系统，拥有强大的生态圈
 - 解释型语言，完美的平台可移植性
 - 支持面向对象和函数式编程
 - 能够通过调用C/C++代码扩展功能
 - 代码规范程度高，可读性强

目前几个比较流行的领域，Python都有用武之地。

 - 云基础设施 - Python / Java / Go
 - DevOps - Python / Shell / Ruby / Go
 - 网络爬虫 - Python / PHP / C++
 - 数据分析挖掘 - Python / R / Scala / Matlab
 - 机器学习 - Python / R / Java / Lisp

作为一名Python开发者，主要的就业领域包括：

- Python服务器后台开发 / 游戏服务器开发 / 数据接口开发工程师
- Python自动化运维工程师
- Python数据分析 / 数据可视化 / 大数据工程师
- Python爬虫工程师
- Python聊天机器人开发 / 图像识别和视觉算法 / 深度学习工程师

给初学者的几个建议：

- Make English as your working language.
- Practice makes perfect.
- All experience comes from mistakes.
- Don't be one of the leeches.
- Either stand out or kicked out.

## [Python语言基础](./Python/Python语言基础/)

### [Python PEP8](./Python/PEP8.md)
    
- Python编码规范

### [初识Python](./Python/Python语言基础/00-初识Python.md)

- Python简介 - Python的历史 / Python的优缺点 / Python的应用领域
- 搭建编程环境 - Windows环境 / Linux环境 / MacOS环境
- 从终端运行Python程序 - Hello, world / print函数 / 运行程序
- 使用IDLE - 交互式环境(REPL) / 编写多行代码 / 运行程序 / 退出IDLE
- 注释 - 注释的作用 / 单行注释 / 多行注释

### [语言元素](./Python/Python语言基础/01-语言元素.md)

- 程序和进制 - 指令和程序 / 冯诺依曼机 / 二进制和十进制 / 八进制和十六进制
- 变量和类型 - 变量的命名 / 变量的使用 / input函数 / 检查变量类型 / 类型转换
- 数字和字符串 - 整数 / 浮点数 / 复数 / 字符串 / 字符串基本操作 / 字符编码
- 运算符 - 数学运算符 / 赋值运算符 / 比较运算符 / 逻辑运算符 / 身份运算符 / 运算符的优先级
- 应用案例 - 华氏温度转换成摄氏温度 / 输入圆的半径计算周长和面积 / 输入年份判断是否是闰年

### [分支结构](./Python/Python语言基础/02-分支结构.md)

- 分支结构的应用场景 - 条件 / 缩进 / 代码块 / 流程图
- if语句 - 简单的if / if-else结构 / if-elif-else结构 / 嵌套的if
- 应用案例 - 用户身份验证 / 英制单位与公制单位互换 / 掷骰子决定做什么 / 百分制成绩转等级制 / 分段函数求值 / 输入三条边的长度如果能构成三角形就计算周长和面积

### [循环结构](./Python/Python语言基础/03-循环结构.md)

- 循环结构的应用场景 - 条件 / 缩进 / 代码块 / 流程图
- while循环 - 基本结构 / break语句 / continue语句
- for循环 - 基本结构 / range类型 / 循环中的分支结构 / 嵌套的循环 / 提前结束程序 
- 应用案例 - 1~100求和 / 判断素数 / 猜数字游戏 / 打印九九表 / 打印三角形图案 / 猴子吃桃 / 百钱百鸡

### [构造程序逻辑](./Python/Python语言基础/04-练习.md)

- 基础练习 - 水仙花数 / 完美数 / 五人分鱼 / Fibonacci数列 / 回文素数 
- 综合练习 - Craps赌博游戏

### [函数和模块的使用](./Python/Python语言基础/05-函数和模块的使用.md)

- 函数的作用 - 代码的坏味道 / 用函数封装功能模块
- 定义函数 - def语句 / 函数名 / 参数列表 / return语句 / 调用自定义函数
- 调用函数 - Python内置函数 /  导入模块和函数
- 函数的参数 - 默认参数 / 可变参数 / 关键字参数 / 命名关键字参数
- 函数的返回值 - 没有返回值  / 返回单个值 / 返回多个值
- 作用域问题 - 局部作用域 / 嵌套作用域 / 全局作用域 / 内置作用域 / 和作用域相关的关键字
- 用模块管理函数 - 模块的概念 / 用自定义模块管理函数 / 命名冲突的时候会怎样（同一个模块和不同的模块）

### [字符串和常用数据结构](./Python/Python语言基础/06-字符串和常用数据结构.md)

- 字符串的使用 - 计算长度 / 下标运算 / 切片 / 常用方法
- 列表基本用法 - 定义列表 / 用下表访问元素 / 下标越界 / 添加元素 / 删除元素 / 修改元素 / 切片 / 循环遍历
- 列表常用操作 - 连接 / 复制(复制元素和复制数组) / 长度 / 排序 / 倒转 / 查找
- 生成列表 - 使用range创建数字列表 / 生成表达式 / 生成器
- 元组的使用 - 定义元组 / 使用元组中的值 / 修改元组变量 / 元组和列表转换
- 集合基本用法 - 集合和列表的区别 /  创建集合 / 添加元素 / 删除元素 /  清空
- 集合常用操作 - 交集 / 并集 / 差集 / 对称差 / 子集 / 超集
- 字典的基本用法 - 字典的特点 / 创建字典 / 添加元素 / 删除元素 / 取值 / 清空
- 字典常用操作 - keys()方法 / values()方法 / items()方法 / setdefault()方法
- 基础练习 - 跑马灯效果 / 列表找最大元素 / 统计考试成绩的平均分 / Fibonacci数列 / 杨辉三角
- 综合案例 - 双色球选号 / 井字棋

### [面向对象编程基础](./Python/Python语言基础/07-面向对象编程基础.md)

- 类和对象 - 什么是类 / 什么是对象 / 面向对象其他相关概念
- 定义类 - 基本结构 / 属性和方法 / 构造器 / 析构器 / \_\_str\_\_方法
- 使用对象 - 创建对象 / 给对象发消息
- 面向对象的四大支柱 - 抽象 / 封装 / 继承 / 多态
- 基础练习 - 定义学生类 / 定义时钟类 / 定义图形类 / 定义汽车类

### [面向对象编程进阶](./Python/Python语言基础/08-面向对象编程进阶.md)

- 属性 - 类属性 / 实例属性 / 属性访问器 / 属性修改器 / 属性删除器 / 使用\_\_slots\_\_
- 类中的方法 - 实例方法 / 类方法 / 静态方法
- 运算符重载 - \_\_add\_\_ / \_\_sub\_\_ / \_\_or\_\_ /\_\_getitem\_\_ / \_\_setitem\_\_ / \_\_len\_\_ / \_\_repr\_\_ / \_\_gt\_\_ / \_\_lt\_\_ / \_\_le\_\_ / \_\_ge\_\_ / \_\_eq\_\_ / \_\_ne\_\_ / \_\_contains\_\_ 
- 类(的对象)之间的关系 - 关联 / 继承 / 依赖
- 继承和多态 - 什么是继承 / 继承的语法 / 调用父类方法 / 方法重写 / 类型判定 / 多重继承 / 菱形继承(钻石继承)和C3算法
- 综合案例 - 工资结算系统 / 图书自动折扣系统 / 自定义分数类

### [图形用户界面和游戏开发](./Python/Python语言基础/09-图形用户界面和游戏开发.md)

- 使用tkinter开发GUI
- 使用pygame三方库开发游戏应用
- “大球吃小球”游戏

### [文件和异常](./Python/Python语言基础/10-文件和异常.md)

- 读文件 - 读取整个文件 / 逐行读取 / 文件路径
- 写文件 - 覆盖写入 / 追加写入 / 文本文件 / 二进制文件
- 异常处理 - 异常机制的重要性 / try-except代码块 / else代码块 / finally代码块 / 内置异常类型 / 异常栈 / raise语句
- 数据持久化 - CSV文件概述 / csv模块的应用 / JSON数据格式 / json模块的应用

### [字符串和正则表达式](./Python/Python语言基础/11-字符串和正则表达式.md)

- 字符串高级操作 - 转义字符 / 原始字符串 / 多行字符串 / in和 not in运算符 / is开头的方法 / join和split方法 / strip相关方法 / pyperclip模块 / 不变字符串和可变字符串 / StringIO的使用
- 正则表达式入门 - 正则表达式的作用 / 元字符 / 转义 / 量词 / 分组 / 零宽断言 /贪婪匹配与惰性匹配懒惰 / 使用re模块实现正则表达式操作（匹配、搜索、替换、捕获）
- 使用正则表达式 - re模块 / compile函数 / group和groups方法 / match方法 / search方法 / findall和finditer方法 / sub和subn方法 / split方法
- 应用案例 - 使用正则表达式验证输入的字符串

### [进程和线程](./Python/Python语言基础/12-进程和线程.md)

- 进程和线程的概念 - 什么是进程 / 什么是线程 / 多线程的应用场景
- 使用进程 - fork函数 / multiprocessing模块 / 进程池 / 进程间通信
- 使用线程 - thread模块 / threading模块 / Thread类 / Lock类 / Condition类 / 线程池

### [网络编程入门](./Python/Python语言基础/13-网络编程入门.md)

- 计算机网络基础 - 计算机网络发展史 / “TCP-IP”模型 / IP地址 / 端口 / 协议 / 其他相关概念
- 网络应用模式 - “客户端-服务器”模式 / “浏览器-服务器”模式
- 基于HTTP协议访问网络资源 - 网络API概述 / 访问URL / requests模块 / 解析JSON格式数据
- Python网络编程 - 套接字的概念 / socket模块 /  socket函数 / 创建TCP服务器 / 创建TCP客户端 / 创建UDP服务器 / 创建UDP客户端 / SocketServer模块

### [网络应用开发](./Python/Python语言基础/14-网络应用开发.md)

- 电子邮件 - SMTP协议 / POP3协议 / IMAP协议 / smtplib模块 / poplib模块 / imaplib模块
- 短信服务 - 调用短信服务网关

### [图像和文档处理](./Python/Python语言基础/15-图像和办公文档处理.md)

- 用Pillow处理图片 - 图片读写 / 图片合成 / 几何变换 / 色彩转换 / 滤镜效果
- 读写Word文档 - 文本内容的处理 / 段落 / 页眉和页脚 / 样式的处理
- 读写Excel文件 - xlrd模块 / xlwt模块
- 生成PDF文件 - pypdf2模块 / reportlab模块

## [Python语言进阶 ](./Python/Python语言进阶)

- [常用数据结构](./Python/Python语言进阶/01-常用数据结构和算法.md)
- [函数的高级用法](./Python/Python语言进阶/02-函数的高级用法.md) - “一等公民” / 高阶函数 / Lambda函数 / 作用域和闭包 / 装饰器
- [面向对象高级知识](./Python/Python语言进阶/03-面向对象高级知识.md) - “三大支柱” / 类与类之间的关系 / 垃圾回收 / 魔术属性和方法 / 混入 / 元类 / 面向对象设计原则 / GoF设计模式
- [迭代器和生成器](./Python/Python语言进阶/04-迭代器和生成器.md) - 相关魔术方法 / 创建生成器的两种方式 / 
- [并发和异步编程](./Python/Python语言进阶/05-并发和异步编程.md) - 多线程 / 多进程 / 异步IO / async和await

## [Python 第三方库](./Python/第三方库)

### [Django](./Python/第三方库/Django)

#### [快速上手](./Python/第三方库/Django/01-快速上手.md)

- Web应用工作原理和HTTP协议
- Django框架概述
- 5分钟快速上手
- 使用视图模板

#### [深入模型](./Python/第三方库/Django/02-深入模型.md)

- 关系型数据库配置
- 管理后台的使用
- 使用ORM完成对模型的CRUD操作
- Django模型最佳实践
- 模型定义参考

#### [静态资源和Ajax请求](./Python/第三方库/Django/03-静态资源和Ajax请求.md)

- AJAX准备知识：JSON / AJAX简介
- jQuery实现AJAX / JS实现AJAX
- AJAX请求如何设置csrf_token
- AJAX上传文件 / AJAX中参数traditional的作用
- 序列化 / SweetAlter插件
- 加载静态资源
- 用Ajax请求获取数据

#### [Django模板系统](./Python/第三方库/Django/04-Django模板系统.md)

- 常见语法：变量相关用{{ }}， 逻辑相关用：{% %}
- 过滤器：default / length / filesizeformat / slice / date / safe / truncatechars / 
         trunatewords / cut / join / timesince / timeuntil / 自定义filter
- Tags：for循环 / if判断 / with / csrf_token / 注释 / 注意事项 
- 母版 / 继承母版 /  块(block) / 组件
- 静态文件相关：{% static %} / {% get_static_prefix %}
- simple_tag / inclusion_tag

#### [Django的View(视图)](./Python/第三方库/Django/05-Django的View.md)

- 一个简单的视图
- CBV和FBV
- 给视图加装饰器：使用装饰器装饰FBV / 使用装饰器装饰CBV
- Request对象和Response对象：方法 / 属性 / 使用
- JsonResponse对象
- Django shortcut functions：render() / redirect()

#### [Django的路由系统](./Python/第三方库/Django/06-Django的路由系统.md)

- URLconfs配置：基本格式 / 参数说明
- 正则表达式详解： 基本配置 / 注意事项 / 补充说明
- 匹配：分组匹配 / 分组命名匹配
- 分组命名匹配：URLconfs匹配的位置 / 捕获的参数永远都是字符串 / 视图函数中指定默认值 / include其他的URLconfs
- 传递额外的参数给视图函数
- 命名URL和URL反向解析
- 命名空间模式

#### [Django ORM相关操作](./Python/第三方库/Django/07-Django ORM相关操作.md)

- 必知必会13方法：all() / filter() / get() / exclude() / values() / values_list() / order_by() / reverse() / distinct() / count() / first() / last() / exists()
- 单表查询之神奇的双下划线
- ForeignKey：正向查找 / 反向查找
- OneToOneField
- ManyToManyField：方法：create() / add() / set() / remove() / clear() / 自己创建第三张表
- 聚合和分组查询：aggregate() 
- F查询和Q查询
- 锁和事务：select_for_update(nowait=False, skip_locked=False)
- Django执行原生SQL：raw()
- Django方法大全
- Django终端打印SQL语句 / 在Python脚本中调用Django环境

#### [Cookie、Session和分页](./Python/第三方库/Django/08-Cookie、Session和分页.md)

- 实现用户跟踪
- cookie和session的关系
- Django框架对session的支持
- 视图函数中的cookie读写操作

#### [Form、ModelForm组件](./Python/第三方库/Django/09-Form和ModelForm组件.md)

- 表单和表单控件
- 常用字段与插件
- 字段校验 / 钩子方法
- 应用bootstrap样式
- 跨站请求伪造和CSRF令牌
- Form和ModelForm
- 表单验证

#### [中间件](./Python/第三方库/Django/10-中间件.md)

- 什么是中间件
- 中间件：process_request / process_response / process_view / process_exception / process_template_response
- Django框架内置的中间件
- 自定义中间件及其应用场景

### [PyQt5](./Python/第三方库/PyQt5)
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
  - [数据库三大范式](./Database/MySQL/数据库三大范式.md)
  
### [NoSQL入门](./Day36-40/39.NoSQL入门.md)
  - NoSQL概述
  - Redis概述
  - Mongo概述

## [Web前端](./Web/)

### [Web前端入门](./Web/Web前端概述.md)

 - 用HTML标签承载页面内容
 - 用CSS渲染页面
 - 用JavaScript处理交互式行为
 - jQuery入门和提高
 - Vue.js入门
 - Element的使用
 - Bootstrap的使用

### [RESTful规范](./Web/RESTful.md)

 - API与用户的通信协议 / 域名 / 版本 / 路径 / method / 过滤 
 - 状态码+自定义状态码 / 错误处理 / 返回结果 / Hypermedia API

## [工具](./Utils/)

### [Docker](./Utils/Docker.md)

- 基本组成 / 容器相关技术简介 / 客户端与守护进程 / 守护进程的配置和操作
- 远程访问 / 容器 / 容器中部署静态网站 / 查看和删除镜像 / 获取和推送镜像
- 构建镜像 / DockerFile指令 / 构建过程 / 容器的网络连接 / 容器的数据卷
- 数据卷容器 / 数据卷的备份和还原
-[Docker入门PDF](./Utils/Docker入门教程.pdf)

### [Git](./Utils/Git基本命令.md)

- 帮助信息 / git配置(全局配置) / 初始化项目 / 查看状态 / 添加文件 / 提交文件 / 查看提交日志
- 查看文件修改前后的区别 / git跟踪rename文件和移动文件 / 删除文件 / 恢复文件 / 恢复提交 
- 重置提交指针 / 查看、创建、切换分支 / 查看两个分支的区别 / 合并分支 / 解决合并冲突 / 重命名和删除分支
- 保存修改进度 / 添加别名 / 全局忽略跟踪文件 / 项目级忽略文件 / 创建远程版本库 / 推送版本库
- 修改远程仓库地址 / 克隆版本库到本地 / 更新本地版本库 / 基于版本库开发自己的版本库 / 添加pull request / 添加贡献者







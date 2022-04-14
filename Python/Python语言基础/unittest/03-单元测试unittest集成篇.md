###### datetime:2022/04/14 15:17

###### author:nzb

## 单元测试：unittest 集成篇

### 一、数据驱动简介

- 为什么需要数据驱动？  
  正例、反例  
  登录：同一个业务逻辑，代码逻辑是不变的，数据有很多组，业务逻辑和数据分离。

### 二、自动化主流的驱动模式介绍

- 数据驱动  
  数据驱动把数据保存excel、csv、yaml、数据库，然后通过改变数据驱动我们的业务逻辑执行，并且得到不同的结果
- 关键字驱动  
  关键字驱动其实是从面向对象的思想触发，它把一些业务逻辑代码封装成一个函数，方法作为一个关键字，然后调用不同的函数组成不同的复杂的业务逻辑

- 数据驱动+关键字驱动

### 三、unittest 的 ddt 数据驱动

- 什么是 ddt ？  
  data driver test, 它可以完美的应用于 unittest 框架实现数据驱动

- ddt 详解  
  它是通过装饰器的方式来调用的
    - 分为类装饰器和函数装饰器
        - `@ddt`：类装饰器，申明当前类使用 DDT 框架
        - `@data`：函数装饰器，用不给测试用例传递数据
        - `@unpack`：函数装饰器，降数据解包，一般用于元组和列表
        - `@file_data`：函数装饰器，用于读取json或yaml文件
    - 用法

```python
    __date__ = "2022/4/14 15:30"
    
    import unittest
    
    from ddt import ddt, data, unpack
    
    @ddt
    class TestDdt(unittest.TestCase):
        """
    
        @data("data1")
            .
            ----------------------------------------------------------------------
            Ran 1 test in 0.000s
    
            OK
            ('data1',)
            {}
    
        @data("data1", "data2")
            ('data1',)
            {}
            ('data2',)
            {}
            ..
            ----------------------------------------------------------------------
            Ran 2 tests in 0.000s
    
            OK
    
        @data(("data1", "data3"), ("data2", "data4"))
            ..
            ----------------------------------------------------------------------
            Ran 2 tests in 0.000s
    
            OK
            (('data1', 'data3'),)
            {}
            (('data2', 'data4'),)
            {}
    
        @data(("data1", "data3"), ("data2", "data4"))
        @unpack
            ('data1', 'data3')
            {}
            ('data2', 'data4')
            {}
            ..
            ----------------------------------------------------------------------
            Ran 2 tests in 0.000s
    
            OK
    
        @data({"a": "data1", "b": "data3"}, {"a": "data2", "b": "data4"})
        @unpack
            ()
            {'a': 'data1', 'b': 'data3'}
            ()
            {'a': 'data2', 'b': 'data4'}
            ..
            ----------------------------------------------------------------------
            Ran 2 tests in 0.000s
    
            OK
    
        @data({"a": "data1", "b": "data3"}, ("data2", "data4"))
        @unpack
            ()
            {'a': 'data1', 'b': 'data3'}
            ('data2', 'data4')
            {}
            ..
            ----------------------------------------------------------------------
            Ran 2 tests in 0.000s
    
            OK
        """
    
        # @data("data1")
        # @data("data1", "data2")
        # @data(("data1", "data3"), ("data2", "data4"))
        # @data({"a": "data1", "b": "data3"}, {"a": "data2", "b": "data4"})
        @data({"a": "data1", "b": "data3"}, ("data2", "data4"))
        @unpack
        def test_01(self, *args, **kwargs):
            """
            测试 test_01
            :return:
            """
            print(args)
            print(kwargs)
    
    if __name__ == '__main__':
        unittest.main()
    
```

- 总结
    - ddt 数据驱动中，测试用例的执行次数是有 @data() 传参的个数决定。传一个值用例执行一次，传多个值，用例执行多次
    - 如果传的是元组(或列表)，那么可以使用 @unpack 解包元组和列表，但是需要注意的是，元组和列表中有多少个值，那么就必须用多少个变量来接收值，或者使用可变长度关键字和关键字参数接收
    - 如果传的是多个字典，那么可以使用 @unpack 解包，但是需要注意的是：用例中的参数的名称和个数必须和字典的 key 保持一致，或者使用可变长度关键字和关键字参数接收
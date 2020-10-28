###### datetime:2020/10/27 16:58
###### author:nzb

# python函数重载机制？
函数重载主要是为了解决两个问题。
- 1。可变参数类型。
- 2。可变参数个数。

另外，一个基本的设计原则是，仅仅当两个函数除了参数类型和参数个数不同以外，其功能是完全相同
的，此时才使用函数重载，如果两个函数的功能其实不同，那么不应当使用重载，而应当使用一个名字
不同的函数。

好吧，那么对于情况 1 ，函数功能相同，但是参数类型不同，python 如何处理？答案是根本不需要处
理，因为 python 可以接受任何类型的参数，如果函数的功能相同，那么不同的参数类型在 python 中
很可能是相同的代码，没有必要做成两个不同函数。

那么对于情况 2 ，函数功能相同，但参数个数不同，python 如何处理？大家知道，答案就是缺省参
数。对那些缺少的参数设定为缺省参数即可解决问题。因为你假设函数功能相同，那么那些缺少的参数
终归是需要用的。

好了，鉴于情况 1 跟 情况 2 都有了解决方案，python 自然就不需要函数重载了。

# 回调函数，如何通信的?

回调函数是把函数的指针(地址)作为参数传递给另一个函数，将整个函数当作一个对象，赋值给调用的函数。

# 闭包延迟

[详情](https://blog.csdn.net/xie_0723/article/details/53925076)

下面这段代码的输出结果将是什么？请解释。
```python
    def multipliers():
        return [lambda x: i *x for i in range(4)]
    print([m(2) for m in multipliers()])
```
上面代码的输出结果是 [6,6,6,6]，不是我们想的 [0,2,4,6]

上述问题产生的原因是python闭包的延迟绑定。这意味着内部函数被调用时，参数的值在闭包内进行查找。
因此，当任何由multipliers()返回的函数被调用时,i的值将在附近的范围进行查找。
那时，不管返回的函数是否被调用，for循环已经完成，i被赋予了最终的值3.

```python
def multipliers():
    for i in range(4):
        yield lambda x: i *x
```

你如何修改上面的 multipliers 的定义产生想要的结果？

```python
def multipliers():
    return [lambda x,i = i: i*x for i in range(4)]

```

# 单例模式

- 装饰器

```python
from functools import wraps

def singleton(cls):
    _instance = {}
    
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return wrapper
```

- 使用基类

```python
class SingletonMeta(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "__instance"):
            setattr(cls, "__instance", super().__new__(cls, *args, **kwargs))
        return getattr(cls, "__instance")

class Foo(SingletonMeta):
    pass
```

- 使用元类

```python
class SingletonMeta(type):
    """自定义元类"""

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "__instance"):
            setattr(cls, "__instance", super().__call__(*args, **kwargs))
        return getattr(cls, "__instance")

class Foo(metaclass=SingletonMeta):
    pass
```

# 请用一行代码实现将 1-N 的整数列表以 3 为单位分组
```python
    N =100
    print ([[x for x in range(1,100)] [i:i+3] for i in range(0,100,3)])
```

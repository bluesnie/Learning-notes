###### datetime:2020/10/27 16:58
###### author:nzb

# 文件操作
现在要处理一个大小为 10 G 的文件，但是内存只有 4 G，如果在只修改 get_lines 函数而其他代码保持不变的情况下，应该如何实现？需要考虑的问题都有那些？
```python
def get_lines():
    with open('file.txt','rb') as f:
        for i in f:
            yield i        
```

# 遍历目录与子目录

抓取.pyc文件
- 第一种
```python
import os
def get_files(dir,suffix):
    res = []
    for root,dirs,files in os.walk(dir):
        for filename in files:
            name,suf = os.path.splitext(filename)
            if suf == suffix:
                res.append(os.path.join(root,filename))
    print(res)
    
get_files("./",'.pyc')
```
- 第二种
```python
import os
def pick(obj):
    if obj.endswith(".pyc"):
        print(obj)
        
def scan_path(ph):
    file_list = os.listdir(ph)
    for obj in file_list:
        if os.path.isfile(obj):
            pick(obj)
        elif os.path.isdir(obj):
            scan_path(obj)
            
if __name__=='__main__':
    path = input('输入目录')
    scan_path(path)
```
- 第三种
```python
from glob import iglob
def func(fp, postfix):
    for i in iglob(f"{fp}/**/*{postfix}", recursive=True):
        print(i)

if __name__ == "__main__":
    postfix = ".pyc"
    func("K:\Python_script", postfix)
```

# 数字字符串转整形

字符串 "123" 转换成 123 ，不使用内置api，例如 int()

- 第一种
```python
def atoi(s):
    num = 0
    for v in s:
        for j in range(10):
            if v == str(j):
                num = num * 10 + j
    return num

```
- 第二种
```python
def atoi(s):
    num = 0
    for v in s:
        num = num * 10 + ord(v) - ord('0')
    return num
```
- 第三种
```python
def atoi(s):
    num = 0
    for v in s:
        t = "%s * 1" % v
        n = eval(t)
        num = num * 10 + n
    return num
```
- 第四种
```python
from functools import reduce
def atoi(s):
    return reduce(lambda num, v: num * 10 + ord(v) - ord('0'), s, 0)
```

# 数字字符串排序
让所有奇数都在偶数前面，而且奇数升序排列，偶数降序排序，如字符串'1982376455',变成'1355798642'

```python
print("".join(sorted('1982376455', key=lambda x: int(x) % 2 == 0 and 20 - int(x) or int(x))))

# 分解
int(x) % 2 == 0 and 20 - int(x)：这是排序偶数，降序排序
int(x)：剩下的奇数升序排序
```

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

# Python的魔法方法

魔法方法就是可以给你的类增加魔力的特殊方法，如果你的对象实现（重载）了这些方法中的某一个，那么这个方法就会在特殊的情况下被Python所调用，
你可以定义自己想要的行为，而这一切都是自动发生的，它们经常是两个下划线包围来命名的（比如 `__init__` , `__len__` ),
Python的魔法方法是非常强的所以了解其使大用方法也变得尤为重要!

- `__init__`： 构造器，当一个实例被创建的时候初始化的方法，但是它并不是实例化调用的第一个方法。
- `__new__`：才是实例化对象调用的第一个方法，它只取下 cls 参数，并把其他参数传给 `__init___`。
- `___new__`： 很少使用，但是也有它适合的场景，尤其是当类继承自一个像元祖或者字符串这样不经常改变的类型的时候。
- `__call__`： 让一个类的实例像函数一样被调用
- `__getitem__`： 定义获取容器中指定元素的行为，相当于 self[key]
- `__getattr__`： 定义当用户试图访问一个不存在属性的时候的行为。
- `__setattr__`： 定义当一个属性被设置的时候的行为
- `__getattribute___`： 定义当一个属性被访问的时候的行为

# 多进程多线程以及协程的理解

这个问题被问的概念相当之大，
- 进程：一个运行的程序（代码）就是一个进程，没有运行的代码叫程序，进程是系统资源分配的最小单
位，进程拥有自己独立的内存空间，所有进程间数据不共享，开销大。
- 线程: cpu调度执行的最小单位，也叫执行路径，不能独立存在，依赖进程存在，一个进程至少有一个线
程，叫主线程，而多个线程共享内存（数据共享，共享全局变量),从而极大地提高了程序的运行效率。
- 协程: 是一种用户态的轻量级线程，协程的调度完全由用户控制。协程拥有自己的寄存器上下文和栈。
协程调度时，将寄存器上下文和栈保存到其他地方，在切回来的时候，恢复先前保存的寄存器上下文和
栈，直接操中栈则基本没有内核切换的开销，可以不加锁的访问全局变量，所以上下文的切换非常快。

# 协程
 
- python asyncio的原理？  
    asyncio 这个库就是使用 python 的 yield 这个可以打断保存当前函数的上下文的机制， 封装好了 selector 摆脱掉了复杂的回调关系
- [单线程+异步 I/O](../../../Python/Python语言基础/12-进程和线程.md#单线程异步I/O) 

# Python异步使用场景有那些
- 1、 不涉及共享资源，获对共享资源只读，即非互斥操作
- 2、 没有时序上的严格关系
- 3、 不需要原子操作，或可以通过其他方式控制原子性
- 4、 常用于IO操作等耗时操作，因为比较影响客户体验和使用性能
- 5、 不影响主线程逻辑

# 多线程竞争
线程是非独立的，同一个进程里线程是数据共享的，当各个线程访问数据资源时会出现竞争状态即：数据几乎同步会被多个线程占用，造成数据混乱，即所谓的线程不安全

那么怎么解决多线程竞争问题？---锁

- 锁的好处： 确保了某段关键代码（共享数据资源）只能由一个线程从头到尾完整地执行能解决多线程资源竞争下的原子操作问题。
- 锁的坏处： 阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行，效率就大大地下降了
- 锁的致命问题: 死锁

# Python的线程同步

- setDaemon(False)

    当一个进程启动之后，会默认产生一个主线程，因为线程是程序执行的最小单位，当设置多线程时，主线程会创建多个子线程，在 Python 中，
默认情况下就是 setDaemon(False) ,主线程执行完自己的任务以后，就退出了，此时子线程会继续执行自己的任务，直到自己的任务结束。

```python
import threading
import time

def thread():
    time.sleep(2)
    print('---子线程结束---')
    
def main():
    t1 = threading.Thread(target=thread)
    t1.start()
    print('---主线程--结束')
    
if __name__ =='__main__':
    main()
    
#执行结果
---主线程--结束
---子线程结束---
```

-  setDaemon（True)
    当我们使用 setDaemon(True) 时，这是子线程为守护线程，主线程一旦执行结束，则全部子线程被强制终止
```python
import threading
import time

def thread():
    time.sleep(2)
    print(’---子线程结束---')

def main():
    t1 = threading.Thread(target=thread)
    t1.setDaemon(True)#设置子线程守护主线程
    t1.start()
    print('---主线程结束---')

if __name__ =='__main__':
    main()

#执行结果
---主线程结束--- #只有主线程结束，子线程来不及执行就被强制结束
```

-  join（线程同步)

    join 所完成的工作就是线程同步，即主线程任务结束以后，进入堵塞状态，一直等待所有的子线程结束以后，主线程再终止。

    当设置守护线程时，含义是主线程对于子线程等待 timeout 的时间将会杀死该子线程，最后退出程序，所以说，如果有 10 个子线程，
    全部的等待时间就是每个 timeout 的累加和，简单的来说，就是给每个子线程一个 timeout 的时间，让他去执行，时间一到，不管任务有没有完成，直接杀死。

    没有设置守护线程时，主线程将会等待timeout的累加和这样的一段时间，时间一到，主线程结束，但是并没有杀死子线程，子线程依然可以继续执行，直到子线程全部结束，程序退出。

```python
import threading
import time

def thread():
    time.sleep(2)
    print('---子线程结束---')

def main():
    t1 = threading.Thread(target=thread)
    t1.setDaemon(True)
    t1.start()
    t1.join(timeout=1)
    #1 线程同步，主线程堵塞1s 然后主线程结束，子线程继续执行
    #2 如果不设置timeout参数就等子线程结束主线程再结束
    #3 如果设置了setDaemon=True和timeout=1主线程等待1s后会强制杀死子线程，然后主线程结束
    print('---主线程结束---')

if __name__=='__main___':
    main()
```

# 锁及其分类

- 定义：锁(Lock)是 python 提供的对线程控制的对象。
- 分类：互斥锁，可重入锁，死锁
    - 死锁
    
        若干子线程在系统资源竞争时，都在等待对方对某部分资源解除占用状态，结果是谁也不愿先解锁，互相干等着，程序无法执行下去，这就是死锁。

    - GIL锁 全局解释器锁（互斥锁）
        - 作用： 限制多线程同时执行，保证同一时间只有一个线程执行，所以cython里的多线程其实是伪多线程！
        所以python里常常使用协程技术来代替多线程，协程是一种更轻量级的线程。
        进程和线程的切换时由系统决定，而协程由我们程序员自己决定，而模块gevent下切换是遇到了耗时操作时才会切换
    - 三者的关系：进程里有线程，线程里有协程。
- 多线程交互访问数据，怎么避免重读？
    
    创建一个已访问数据列表，用于存储已经访问过的数据，并加上互斥锁，在多线程访问数据的时候先查看数据是否在已访问的列表中，若已存在就直接跳过。
    
- 什么是线程安全，什么是互斥锁？

    每个对象都对应于一个可称为’互斥锁‘的标记，这个标记用来保证在任一时刻，只能有一个线程访问该对象。
    
    同一进程中的多线程之间是共享系统资源的，多个线程同时对一个对象进行操作，一个线程操作尚未结束，另一线程已经对其进行操作，
    导致最终结果出现错误，此时需要对被操作对象添加互斥锁，保证每个线程对该对象的操作都得到正确的结果。

# 同步、异步、阻塞、非阻塞

- 同步： 多个任务之间有先后顺序执行，一个执行完下个才能执行。
- 异步： 多个任务之间没有先后顺序，可以同时执行，有时候一个任务可能要在必要的时候获取另一个同时执行的任务的结果，这个就叫回调！
- 阻塞： 如果卡住了调用者，调用者不能继续往下执行，就是说调用者阻塞了。
- 非阻塞： 如果不会卡住，可以继续执行，就是说非阻塞的。
- 同步异步相对于多任务而言，阻塞非阻塞相对于代码执行而言。

# 僵尸进程和孤儿进程及怎么避免僵尸进程？

- 孤儿进程： 父进程退出，子进程还在运行的这些子进程都是孤儿进程，孤儿进程将被 init 进程（进程号为 1 ）所收养，并由 init 进程对他们完成状态收集工作。
- 僵尸进程： 进程使用fork 创建子进程，如果子进程退出，而父进程并没有调用 wait 获 waitpid 获取子进程的状态信息，那么子进程的进程描述符仍然保存在系统中的这些进程是僵尸进程。
- 避免僵尸进程的方法：
    - 1.fork 两次用孙子进程去完成子进程的任务
    - 2.用 wait() 函数使父进程阻塞
    - 3.使用信号量，在 signal handler 中调用 waitpid , 这样父进程不用阻塞
    
# IO密集型和CPU密集型区别？
- IO密集型：系统运行，大部分的状况是CPU在等 I/O（硬盘/内存）的读/写。
- CPU密集型：大部分时间用来做计算，逻辑判断等 CPU 动作的程序称之 CPU 密集型。
    
# python中进程与线程的使用场景？

- 多进程适合在 CPU 密集操作（ cpu 操作指令比较多，如位多的的浮点运算）。
- 多线程适合在 IO 密性型操作（读写数据操作比多的的，比如爬虫）

# 线程是并发还是并行，进程是并发还是并行？
- 并发是指一个处理器同时处理多个任务。
- 并行是指多个处理器或者是多核的处理器同时处理多个不同的任务。
- 线程是并发，进程是并行;
- 进程之间互相独立，是系统分配资源的最小单位，同一个进程中的所有线程共享资源。
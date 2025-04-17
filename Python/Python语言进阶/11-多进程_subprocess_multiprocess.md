###### datetime:2025/04/15 16:28

###### author:nzb
    
# Python多进程 - subprocess & multiprocess 

## 目录
- subprocess
  - 1.1. run(), 阻塞调用
    - 1.1.1. shell选项
    - 1.1.2. 获取输出
    - 1.1.3. check选项
  - 1.2. call(), 旧版本函数
  - 1.3. Popen, 非阻塞
    - 1.3.1. 管理子进程（通信）
      - sys.stdin
      - 示例
- multiprocess
  - 2.1. 创建子进程
    - 2.1.1. 直接使用Process模块创建进程
    - 2.1.2. 继承的形式创建进程
    - 2.1.3. Process模块方法介绍
  - 2.2. 进程管理
    - 2.2.1. 僵尸进程（有害）
    - 2.2.2. 孤儿进程（无害）
    - 2.2.3. 守护进程
  - 2.3. 进程同步
  - 2.4. 进程池
  - 2.5. multiprocessing.Queue

- 两个模块的对比：
  - `multiprocess`: 是同一个代码中通过多进程调用其他的模块（也是自己写的）
  - `subprocess`: 直接调用外部的二进制程序，而非代码模块

## [subprocess](https://docs.python.org/3/library/subprocess.html)

> 该模块主要用于调用子程序。

### 1.1. run(), 阻塞调用

```python
subprocess.call("mv abc", shell=True) # depressed after python3.5
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False) -> subprocess.CompletedProcess
```

`run/call()` 函数均为**主进程阻塞**执行子进程，直到子进程调用完成返回；

#### 1.1.1. shell选项

如果 `shell=True` ，则将通过`shell`执行指定的命令。如果你希望方便地访问其他`shell`功能，如`shell`管道、文件名通配符、环境变量扩展和扩展，例如 `~` 到用户的主目录，这会很有用。

```python
# subprocess.run("ls | grep xxx".split(" "))
# 以上语句，无法获取到想要的结果，因为subprocess本身并不执行 `|` 管道符
subprocess.run("ls | grep xxx".split(" "), shell=True)
```

#### 1.1.2. 获取输出

```python
proc = subprocess.run(['uname', '-r'], stdout=su.PIPE)
print(proc.returncode)
print(proc.stdout)
```

#### 1.1.3. check选项

当设置 `check=True` 时，如果`returncode`返回非0，则抛出异常 `CalledProcessError` 。

> If check is True and the exit code was non-zero, it raises a CalledProcessError . The CalledProcessError object will have the return code in the returncode attribute, and output & stderr attributes if those streams were captured.

### 1.2. call(), 旧版本函数

如果你不得不在`Python3.5`之前的版本运行，那么你可以选用以下几种方式，来替代 `subprocess.run()` 的调用。

```python
subprocess.call(args, *, stdin=None, stdout=None, stderr=None, shell=False, cwd=None, timeout=None, **other_popen_kwargs)
```

如果希望在 `call()` 处理过程中使用 `check=True` ，则直接使用 `check_call()` 函数，它等效于 `run(..., check=True) `。

> Note Do not use stdout=PIPE or stderr=PIPE with ## this function. The child process will block if it generates enough output to a pipe to fill up the OS pipe buffer as the pipes are not being read from.

那么如何获取到 `stdout` 呢？

```python
subprocess.check_output(args, *, stdin=None, stderr=None, shell=False, cwd=None, encoding=None, errors=None, universal_newlines=None, timeout=None, text=None, **other_popen_kwargs)
```

相当于: `run(..., check=True, stdout=PIPE).stdout`
也可以捕获到 `stderr`，使用 `stderr=subprocess.STDOUT`:

```python
>>> subprocess.check_output(
...     "ls non_existent_file; exit 0",
...     stderr=subprocess.STDOUT,
...     shell=True)
'ls: non_existent_file: No such file or directory\n'
```

### 1.3. [Popen](https://docs.python.org/3/library/subprocess.html#popen-constructor), 非阻塞

- 简单示例

```python
import subprocess as sub

>>> ls = sub.Popen(['ls'], stdout=sub.PIPE)
>>> for f in ls.stdout: print(f)
...
b'fileA.txt\n'
b'fileB.txt\n'
b'ls.txt\n'

>>> ex = sub.Popen(['ex', 'test.txt'], stdin=sub.PIPE)
>>> ex.stdin.write(b'i\nthis is a test\n.\n')
>>> ex.stdin.flush()  # 强制刷新缓冲区
19
>>> ex.std.write(b'wq\n')
3
>>> ex.stdin.flush()  # 强制刷新缓冲区
>>> ex.stdin.close()  
>>> ex.wait()  # 等待进程结束
0
```

注意： `shell=True` 选项会开启`Windows`控制台执行命令，这会引发一个问题——`Windows`中的控制台命令是后台执行的！所以当`python`调用 `Popen.terminate()` 时，只是关闭了控制台，控制台命令却仍在后台执行（直至结束）。所以，如需关闭时同时关闭命令，请不要使用 `Popen(str_cmd, shell=True)`，用 `Popen(str_cmd.split(' ')` 代替。

```python
p.poll()  # 检查进程是否终止，如果终止返回returncode，否则返回None
p.wait(timeout)  # 等待子进程终止(阻塞父进程)
p.communicate(input,timeout)  # 和子进程交互，发送和读取数据(阻塞父进程)
p.send_signal(singnal)  # 发送信号到子进程
p.terminate()  # 停止子进程,也就是发送SIGTERM信号到子进程
p.kill()  # 杀死子进程。发送SIGKILL信号到子进程
```

创建`Popen`对象后，**主程序不会自动等待子进程完成**。

以上三个成员函数都可以用于等待子进程返回：`while`循环配合`Popen.poll()`、`Popen.wait()`、`Popen.communicate()`。**由于后面二者都会阻塞父进程，所以无法实时获取子进程输出**，而是等待子进程结束后一并输出所有打印信息。另外，`Popen.wait()`、`Popen.communicate()`分别将输出存放于管道和内存，前者容易超出默认大小而导致死锁，因此不推荐使用。

注意：`p.communicate(stdin="xxx")` 该函数会终止子程序（因为其是阻塞的，当父程序解除阻塞时，意味着子程序已经结束了）。所以，如果你的子程序是 `while...` 或者 `for line in sys.stdin` 时，你会发现**子程序意外的结束了**，而不是在循环中等待。

#### 1.3.1. 管理子进程（通信）

`Popen`类具有三个与输入输出相关的属性：`Popen.stdin` , `Popen.stdout` 和 `Popen.stderr` ，分别对应子进程的标准输入/输出/错误。它们的值可以是`PIPE`、文件描述符(正整数)、文件对象或`None`：
- `PIPE`表示创建一个连接子进程的新管道，默认值`None`, 表示不做重定向。
- 子进程的文件句柄可以从父进程中继承得到。
- 仅 `stderr` 可以设置为 `STDOUT` ，表示将子进程的标准错误重定向到标准输出。

```python
child1 = subprocess.Popen(["ls","-l"], stdout=subprocess.PIPE)
child2 = subprocess.Popen(["wc"], stdin=child1.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
out = child2.communicate()
```

其中，`subprocess.PIPE`为文本流提供一个缓存区，`child1`的`stdout`将文本输出到缓存区；随后`child2`的`stdin`从该`PIPE`读取文本，`child2`的输出文本也被存放在`PIPE`中，而标准错误信息则重定向到标准输出；最后，`communicate()`方法从`PIPE`中读取`child2`子进程的标准输出和标准错误。

**注意**：`subprocess.stdxxx` 操作`bytes`字节，而 `sys.stdin` 则是`string`。


##### sys.stdin

`Python`的`sys`模块定义了标准输入/输出/错误：

```python
sys.stdin  # 标准输入
sys.stdout # 标准输出
sys.stderr # 标准错误信息
```

以上三个对象类似于文件流，因此可以使用`readline()`和`write()`方法进行读写操作。也可以使用`print()`，等效于`sys.stdout.write()`。

需要注意的是，除了直接向控制台打印输出外，标准输出/错误的打印存在缓存，为了实时输出打印信息，需要执行

```python
sys.stdout.flush()
sys.stderr.flush()
```

读取 `sys.stdin` 的方式：

```python
for line in sys.stdin:
    print(type(line))  # string
    ...
```

##### 示例

前面提到，如果子程序只是调用一次，并获取其输出状态，可以使用 `p.communicate(stdin, timeout)` 结合 `p.returncode` 实现。

但如果你的程序想实现类似 `TCP-C/S` 式的持续通信服务，这里提供一个`Demo`:

父程序：

```python
proc = subprocess.Popen(command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE)
                        # stderr=subprocess.STDOUT)

# try:
while proc.poll() is None:  # 持续输入
    str_input = input("Please Input a path: ")
    if str_input == "quit":
        break

    bytes_path = f"/home/brt/{str_input}.jpg\n".encode()  # 注意这里需要\n换行符
    proc.stdin.write(bytes_path)  # 需要使用bytes
    proc.stdin.flush()
    # proc.communicate(stdin=str_input, timout=5)

    bytes_state = proc.stdout.readline()  # bytes
    if bytes_state == b"ok\n":
        print("Well Done.")

# except subprocess.TimeoutExpired:
#     print("子程序Timout未响应...")
#     break

# if proc.poll() is None:  # communicate()超时时，子程序可能未退出
#     proc.kill()
```

子程序：

```python
for path_save in sys.stdin:  # 持续读取
    path_save = path_save.strip()  # 删除多余的换行符
    img = grabclipboard_byQt(cb)
    # sys.stderr.write(">>>", img)
    if img:
        save_clipboard_image(img, path_save)
        str_pipe = "ok"
    else:
        str_pipe = ""

    # 以下内容用于写入stdout管道，向父程序反馈
    # sys.stdout.write(str_pipe + "\n")  # 必须添加换行符
    print(str_pipe)
    sys.stdout.flush()  # 及时清空缓存
```

## [multiprocess](https://docs.python.org/3/library/multiprocessing.html)


### 2.1. 创建子进程

#### 2.1.1. 直接使用Process模块创建进程

```python
Process([group [, target [, name [, args [, kwargs]]]]])
```

- `group`: 参数未使用，值始终为`None`
- `target`: 表示调用对象，即子进程要执行的任务
- `name`: 子进程的名称
- `args`: 调用对象的位置参数元组，`args=(1,2,'egon',)`
- `kwargs`: 调用对象的字典,`kwargs=`

```python
from multiprocessing import Process

def func():
    print("子进程正在运行")
    time.sleep(1)
    print("子进程ID>>>", os.getpid())
    print("子进程的父进程ID>>>", os.getppid())

if __name__ == '__main__':
    p = Process(target=func,)
    p.start()

    print("父进程ID>>>", os.getpid())
    print("父进程的父进程ID>>>", os.getppid())

    p.join()
    print("主进程执行完毕!")

"""
父进程ID>>> 18689
父进程的父进程ID>>> 15674
子进程正在运行
子进程ID>>> 18690
子进程的父进程ID>>> 18689
主进程执行完毕!
"""
```

#### 2.1.2. 继承的形式创建进程

```python
class MyProcess(Process):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        print("子进程的名字是>>>", self.name)
        print("子进程ID>>>", os.getpid(), "子进程的父进程ID>>>", os.getppid())
        time.sleep(2)
        print("子进程结束")


if __name__ == '__main__':
    p = MyProcess(name="子进程01")
    p.start()
    print("主进程ID>>>", os.getpid(), "主进程的父进程ID>>>", os.getppid())
    print("p.name:", p.name, "\np.pid:", p.pid)
    print("p.is_alive:", p.is_alive())

    # p.terminate()  # 给操作系统发送一个关闭进程p1的信号,让操作系统去关闭它
    p.join()
    print("主进程结束")

"""
主进程ID>>> 18425 主进程的父进程ID>>> 15674
p.name: 子进程01 
p.pid: 18426
p.is_alive: True
子进程的名字是>>> 子进程01
子进程ID>>> 18426 子进程的父进程ID>>> 18425
子进程结束
主进程结束
"""

# 执行p.terminate()后
"""
主进程ID>>> 18505 主进程的父进程ID>>> 15674
p.name: 子进程01 
p.pid: 18506
p.is_alive: True
主进程结束
"""
```

#### 2.1.3. Process模块方法介绍

- `p.start()`: 启动进程，并调用该子进程中的`p.run()`
- `p.run()`: 进程启动时运行的方法，正是它去调用`target`指定的函数
- `p.terminate()`: 强制终止进程`p`，不会进行任何清理操作，如果`p`创建了子进程，该子进程就成了僵尸进程，使用该方法需要特别小心这种情况。如果p还保存了一个锁那么也将不会被释放，进而导致死锁
- `p.is_alive()`: 如果p仍然运行，返回`True`
- `p.join([timeout])`: 主线程等待`p`终止（强调：是主线程处于等的状态，而`p`是处于运行的状态）

### 2.2. 进程管理

#### 2.2.1. [僵尸进程（有害）](../../C++/多进程/02-孤儿进程和僵尸进程.md)
#### 2.2.2. [孤儿进程（无害）](../../C++/多进程/02-孤儿进程和僵尸进程.md)
#### 2.2.3. 守护进程

使用平常的方法时,**子进程是不会随着主进程的结束而结束,只有当主进程和子进程全部执行完毕后,程序才会结束**.但是,如果我们的需求是: 主进程执行结束,由该主进程创建的子进程必须跟着结束. 这时,我们就需要用到守护进程了.

主进程创建守护进程:

- 守护进程会在主进程代码执行结束后就终止
- 守护进程内无法再开启子进程,否则抛出异常: `AssertionError: daemonic processes are not allowed to have children`

需要注意的是: 进程之间是相互独立的,主进程代码运行结束,守护进程随机终止.

```python
class Myprocess(Process):
    def __init__(self,person):
        super().__init__()
        self.person = person

    def run(self):
        print("这个人的ID号是:%s" % os.getpid())
        print("这个人的名字是:%s" % self.name)
        time.sleep(3)
        print("这个人的父亲的ID号是:%s" % os.getppid()) 

if __name__ == '__main__':
    p = Myprocess('李华')
    p.daemon=True #一定要在p.start()前设置,设置p为守护进程,禁止p创建子进程,并且父进程代码执行结束,p即终止运行
    p.start()
    # time.sleep(1) # 在sleep时linux下查看进程id对应的进程ps -ef|grep id
    print('主进程执行完毕!')

"""
这个人的ID号是:18224
这个人的名字是:Myprocess-1
主进程执行完毕!
"""
```

#### 2.3. [进程同步](./12-多进程扩展-进程同步-信号量-事件-队列.md)
#### 2.4. [进程池](./13-多进程扩展-管道-数据共享-进程池.md)
#### 2.5. [multiprocessing.Queue](https://docs.python.org/3/library/multiprocessing.html#queue)

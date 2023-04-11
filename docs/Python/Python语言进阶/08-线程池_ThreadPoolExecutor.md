###### datetime:2023/03/23 10:10

###### author:nzb

## 线程池 ThreadPoolExecutor

从`Python3.2`开始，标准库为我们提供了 `concurrent.futures` 模块，它提供了 `ThreadPoolExecutor` (线程池)和 `ProcessPoolExecutor` (进程池)两个类。

相比 `threading` 等模块，该模块通过 `submit` 返回的是一个 `future` 对象，它是一个未来可期的对象，通过它可以获取某一个线程执行的状态或者某一个任务执行的状态及返回值：

- 主线程可以获取某一个线程（或者任务的）的状态，以及返回值。
- 当一个线程完成的时候，主线程能够立即知道。

### 基础语法

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed


def action(second):
    print(second)
    time.sleep(second)
    return second


lists = [4, 5, 2, 3]

# 创建一个最大容纳数量为2的线程池
pool = ThreadPoolExecutor(max_workers=2)

# 通过submit提交执行的函数到线程池中
all_task = [pool.submit(action, i) for i in lists]

# 通过result来获取返回值
result = [i.result() for i in all_task]
print(f"result:{result}")

print("----complete-----")
# 线程池关闭
pool.shutdown()
```

```text
4
5
2
3
result:[4, 5, 2, 3]
----complete-----
```

### 使用上下文管理器

可以通过 `with` 关键字来管理线程池，当线程池任务完成之后自动关闭线程池。

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed


def action(second):
    print(second)
    time.sleep(second)
    return second


lists = [4, 5, 2, 3]
all_task = []
with ThreadPoolExecutor(max_workers=2) as pool:
    for second in lists:
        all_task.append(pool.submit(action, second))

    result = [i.result() for i in all_task]
    print(f"result:{result}")
```

```text
4
5
2
3
result:[4, 5, 2, 3]
```

### 等待所有主线程完成

在需要返回值的场景下，主线程需要等到所有子线程返回再进行下一步，阻塞在当前。比如下载图片统一保存，这时就需要在主线程中一直等待，使用wait方法完成。

`wait(fs, timeout=None, return_when=ALL_COMPLETED)`

wait 接受三个参数：

- fs: 表示需要执行的序列
- timeout: 等待的最大时间，如果超过这个时间即使线程未执行完成也将返回
- return_when：表示wait返回结果的条件，默认为 `ALL_COMPLETED` 全部执行完成再返回，可选 `FIRST_COMPLETED`

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed

lists = [4, 5, 2, 3]
all_task = []
with ThreadPoolExecutor(max_workers=2) as pool:
    for second in lists:
        all_task.append(pool.submit(action, second))

    # 主线程等待所有子线程完成
    wait(all_task, return_when=ALL_COMPLETED)
    print("----complete-----")
```

```text
4
5
2
3
----complete-----
```

### 等待第一个主线程完成

wait 方法可以设置等待第一个子线程返回就继续执行，表现为主线程在第一个线程返回后便不会阻塞，继续执行下面的操作。

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed


def action(second):
    print(second)
    time.sleep(second)
    return second


lists = [4, 5, 2, 3]
all_task = []
with ThreadPoolExecutor(max_workers=2) as pool:
    for second in lists:
        all_task.append(pool.submit(action, second))

    # 主线程等待第一个子线程完成
    wait(all_task, return_when=FIRST_COMPLETED)
    print("----complete-----")
```

```text
4
5
2
----complete-----
3
```

因为result方法是阻塞的，所以流程会在`result`这里阻塞直到所有子线程返回，相当于 `ALL_COMPLETED` 方法。

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed


def action(second):
    print(second)
    time.sleep(second)
    return second


lists = [4, 5, 2, 3]
all_task = []
with ThreadPoolExecutor(max_workers=2) as pool:
    for second in lists:
        all_task.append(pool.submit(action, second))

    # 主线程等待第一个子线程完成
    wait(all_task, return_when=FIRST_COMPLETED)
    print("----first complete-----")

    result = [i.result() for i in all_task]
    print(f"result:{result}")
    print("----complete-----")
```

```text
4
5
2
----first complete-----
3
result:[4, 5, 2, 3]
----complete-----
```

### 返回及时处理

如果不需要等待所有线程全部返回，而是每返回一个子线程就能够处理，那么就可以使用`as_completed`获取每一个线程的返回结果。

`as_completed()` 方法是一个生成器，在没有任务完成的时候，会一直阻塞，除非设置了 `timeout`。当有某个任务完成的时候，会 `yield` 这个任务， 就能执行 `for`
循环下面的语句，然后继续阻塞住，循环到所有的任务结束。同时，先完成的任务会先返回给主线程。

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed


def action(second):
    print(second)
    time.sleep(second)
    return second


lists = [4, 5, 2, 3]
all_task = []
with ThreadPoolExecutor(max_workers=2) as pool:
    for second in lists:
        all_task.append(pool.submit(action, second))

    for future in as_completed(all_task):
        print(f"{future.result()} 返回")

    print("----complete-----")
```

```text
4
5
2
4 返回
3
5 返回
2 返回
3 返回
----complete-----
```

### map

`map` 方法是对序列中每一个元素都执行 `action` 方法，主要有两个特点：

- 不需要将任务`submit`到线程池
- 返回结果的顺序和元素的顺序相同，即使子线程先返回也不会获取结果

`map(fn, *iterables, timeout=None)`

- fn： 第一个参数 fn 是需要线程执行的函数；
- iterables：第二个参数接受一个可迭代对象；
- timeout： 第三个参数 `timeout` 跟 `wait()` 的 `timeout` 一样，但由于 `map` 是返回线程执行的结果，如果 `timeout`小于线程执行时间会抛异常 `TimeoutError`。

```python
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed


def action(second):
    print(second)
    time.sleep(second)
    return second


lists = [5, 1, 2, 3]
with ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(action, lists):
        print(f"{result} 返回")
```

```text
5
1
2
3
5 返回
1 返回
2 返回
3 返回
```

可以看出返回结果和列表的结果一致，即使第2个元素只需要1s就能返回，也还是等待第一个5s线程返回只有才有结果。
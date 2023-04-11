# ZeroMQ

## 1、zmq套接字

- 创建和销毁套接字：`zmq.socket()`, `zmq.close()`
- 配置和读取套接字：`zmq.setsockopt()`, `zmq.getsockopt()`
- 为套接字建立连接：`zmq.bind()`, `zmq.connect()`
- 发送和接收消息：  `zmq.send()`, `zmq.recv()`

> **注**：
> - 使用`zmq.bind()`连接的节点称之为服务端，它有着一个较为固定的网络地址；
> - 使用`zmq.connect()`连接的节点称为客户端，其地址不固定。

## 2、zmq消息模式

主要有三种常用模式

- `req/rep`(请求答复模式)：主要用于远程调用及任务分配等。
- `pub/sub`(订阅模式)： 主要用于数据分发。
- `push/pull`(管道模式)： 主要用于多任务并行。

## 3、zmq内置的有效绑定对

- `PUB` and `SUB`
- `REQ` and `REP`
- `REQ` and `XREP`
- `XREQ` and `REP`
- `XREQ` and `XREP`
- `XREQ` and `XREQ`
- `XREP` and `XREP`
- `PUSH` and `PULL`
- `PAIR` and `PAIR`

## 4、具体消息模式举例

#### (1)、`req/rep`(请求/答复模式)

一对一模式，一问一答

- server服务端

```python
import zmq

context = zmq.Context(io_threads=5)  # 5个线程
socket = context.socket(zmq.REP)  # 设置socket的类型，zmq.REP答复
socket.bind("tcp://*:15000")  # 绑定服务端的IP和端口

while True:  # 循环接收客户端发来的消息
    message = socket.recv()  # 接收客户端发送来的消息，注：是byte类型
    print(message)
    socket.send_string("copy!")  # 再发回客户端消息
# 结果：客户单每请求一次就打印一次消息体
# b'request'
# b'request'
# b'request'
# b'request'
```

- client客户端

```python
import zmq, sys, threading

context = zmq.Context()
socket = context.socket(zmq.REQ)  # 设置socket类型，请求端 socket.connect("tcp://localhost:15000") #连接服务端的IP和端口
socket.connect("tcp://127.0.0.1:15000")
lock = threading.Lock()

while True:
    data = input("input your request:")
    if data == "q":
        sys.exit()
    # 客户端如果一个进程有多线程使用，需要使用锁，保证send和recv配对使用，否则会报错
    with lock:
        socket.send_string(data)  # 向服务端发送消息 message=socket.recv()              #接收服务端返回的消息，注：是byte类型 print(message)
        recv_msg = socket.recv_string()
        print(f"recv msg: {recv_msg}")
"""
结果：每输入请求一次，就得到服务端的一次返回 input your data:123 b'copy!'
input your data:456 b'copy!'
"""
```

- `send()、recv()`扩展

| 发送 | 接收 | 发送数据结构 | 返回数据结构 |  说明 |
| ----- | ----- | ----- |----- |----- |
| send() | recv() | `bytes, Frame, memoryview` | `bytes, Frame, memoryview` | - |
| send_string() | recv_string() | `str` | `str` | - |
| send_pyobj() | recv_pyobj() | `Python object` | `Python object` | - |
| send_json() | recv_json() | `Python object` | `List, str, int, float, Dict` | - |
| send_serialized() | recv_serialized() | `The message to be sent. Can be any object serializable by serialize.` | `自定义反序列化函数` | 自定义序列化函数 |

> 注意：`zmq.error.ZMQError: Operation cannot be accomplished in current state`
>
> zmq模式为`zmq.REP`。在这种模式下，我们的程序必须要遵守`recv()`和`send()`配对使用的编程模式。
>
> 也就是说，在服务程序中，必须要有完整的`recv()`和`send()`成对出现。同理，在客户端程序中，`send()`后，也要有`recv()`。

#### (2)、`pub/sub`(订阅模式)

一对多模式 一个发布者，多个订阅者，订阅者可以通过设置过滤器过滤数据。

- Publisher发布者

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:15000")
while True:
    data = input("input your data:")
    print(data)
    socket.send_string(data)
"""
结果：循环提示输入数据，当输入一次，就发送一次到订阅者 input your data:123 123 input your data:456 456 input your data:789 789 input your data:
"""
```

- Subscriber订阅者

```python
import sys
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:15000")

socket.setsockopt_string(zmq.SUBSCRIBE, '')
# 或者：
socket.setsockopt_string(zmq.SUBSCRIBE, '123')  # 表示只过滤出收到消息为'123'的消息
# 或者： 
socket.subscribe('topic')  # 订阅一个主题, 表示只过滤出收到消息为'topic'的消息

while True:
    message = socket.recv()
    print(message)
"""
结果：发布者每发布一次，都能订阅到 b'123'
b'456'
b'789'
"""
```

#### (3)、`push/pull`(管道模式)

管道是单向的，从PUSH端单向的向PULL端单向的推送数据流。 由三部分组成，push进行数据推送，work进行数据缓存，pull进行数据竞争获取处理。

> 区别于Publish-Subscribe, 管道模式存在一个数据缓存和处理负载。 当连接被断开，数据不会丢失，重连后数据继续发送到对端。

- 推送端

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)  
# 设置socket类型PUSH推送 
socket.bind("tcp://*:5557")       #绑定IP和端口

while True:
    data = input("input your data:")
    socket.send_string(data)
"""
input your data:123 
input your data:456 
input your data:789
"""
```

- worker端

```python
import zmq

context = zmq.Context()
socket_receive = context.socket(zmq.PULL)  
# 设置socket类型PULL拉取推送端的消息 
socket_receive.connect("tcp://localhost:5557")
# 连接推送端IP和端口
socket_sender = context.socket(zmq.PUSH)  
# 再设置一个socket类型PUSH推送 
socket_sender.connect("tcp://localhost:5558")

# 连接IP和端口向其推送消息

while True:
    data = socket_receive.recv_string()  # 拉取接收消息 print(data)
    socket_sender.send_string(data)  # 再将消息推送出去
"""
123 
456 
789
"""
```

- 拉取端

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)  
# 设置socket类型PULL拉取消息 
socket.bind("tcp://*:5558")        
#绑定IP和端口去拉取消息 
while True:
    message = socket.recv_string()
    print(message)
"""
123 
456 
789
"""
```
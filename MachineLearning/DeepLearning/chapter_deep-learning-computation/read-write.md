###### datetime:2024/08/10 15:14

###### author:nzb

本项目源于《[动手学深度学习](https://github.com/d2l-ai/d2l-zh)》，添加了一些自己的学习笔记，方便搜索查阅。正版GitHub地址：https://github.com/d2l-ai/d2l-zh

# 读写文件

到目前为止，我们讨论了如何处理数据，
以及如何构建、训练和测试深度学习模型。
然而，有时我们希望保存训练的模型，
以备将来在各种环境中使用（比如在部署中进行预测）。
此外，当运行一个耗时较长的训练过程时，
最佳的做法是定期保存中间结果，
以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。
因此，现在是时候学习如何加载和存储权重向量和整个模型了。

## (**加载和保存张量**)

对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。
这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。

```python
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

我们现在可以将存储在文件中的数据读回内存。

```python
#@tab pytorch
x2 = torch.load('x-file')
x2
```

我们可以[**存储一个张量列表，然后把它们读回内存。**]

```python
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

我们甚至可以(**写入或读取从字符串映射到张量的字典**)。
当我们要读取或写入模型中的所有权重时，这很方便。

```python
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

## [**加载和保存模型参数**]

保存单个权重向量（或其他张量）确实有用，
但是如果我们想保存整个模型，并在以后加载它们，
单独保存每个向量则会变得很麻烦。
毕竟，我们可能有数百个参数散布在各处。
因此，深度学习框架提供了内置函数来保存和加载整个网络。
需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型。
例如，如果我们有一个3层多层感知机，我们需要单独指定架构。
因为模型本身可以包含任意代码，所以模型本身难以序列化。
因此，为了恢复模型，我们需要用代码生成架构，
然后从磁盘加载参数。
让我们从熟悉的多层感知机开始尝试一下。


```python
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

接下来，我们[**将模型的参数存储在一个叫做“mlp.params”的文件中。**]

```python
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，我们[**实例化了原始多层感知机模型的一个备份。**]
这里我们不需要随机初始化模型参数，而是(**直接读取文件中存储的参数。**)

```python
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

由于两个实例具有相同的模型参数，在输入相同的`X`时，
两个实例的计算结果应该相同。
让我们来验证一下。

```python
#@tab pytorch, paddle
Y_clone = clone(X)
Y_clone == Y
```

## 小结

* `save`和`load`函数可用于张量对象的文件读写。
* 我们可以通过参数字典保存和加载网络的全部参数。
* 保存架构必须在代码中完成，而不是在参数中完成。

## 练习

1. 即使不需要将经过训练的模型部署到不同的设备上，存储模型参数还有什么实际的好处？
1. 假设我们只想复用网络的一部分，以将其合并到不同的网络架构中。比如想在一个新的网络中使用之前网络的前两层，该怎么做？
1. 如何同时保存网络架构和参数？需要对架构加上什么限制？
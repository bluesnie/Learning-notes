###### datetime:2024/07/22 19:38

###### author:nzb

本项目源于《[动手学深度学习](https://github.com/d2l-ai/d2l-zh)》，添加了一些自己的学习笔记，方便搜索查阅。正版GitHub地址：https://github.com/d2l-ai/d2l-zh

# 线性回归的简洁实现

在过去的几年里，出于对深度学习强烈的兴趣，
许多公司、学者和业余爱好者开发了各种成熟的开源框架。
这些框架可以自动化基于梯度的学习算法中重复性的工作。
在《线性回归的从零开始实现》中，我们只运用了：
（1）通过张量来进行数据存储和线性代数；
（2）通过自动微分来计算梯度。
实际上，由于数据迭代器、损失函数、优化器和神经网络层很常用，
现代深度学习库也为我们实现了这些组件。

本节将介绍如何(**通过使用深度学习框架来简洁地实现**)
 《线性回归的从零开始实现》中的(**线性回归模型**)。

## 生成数据集

与《线性回归的从零开始实现》中类似，我们首先[**生成数据集**]。


```python
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```python
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 读取数据集

我们可以[**调用框架中现有的API来读取数据**]。
我们将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。
此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```python
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```python
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

使用`data_iter`的方式与我们在《线性回归的从零开始实现》中使用`data_iter`函数的方式相同。为了验证是否正常工作，让我们读取并打印第一个小批量样本。
与《线性回归的从零开始实现》不同，这里我们使用`iter`构造Python迭代器，并使用`next`从迭代器中获取第一项。

```python
#@tab all
next(iter(data_iter))
# [tensor([[-1.3116, -0.3062],
#          [-1.5653,  0.4830],
#          [-0.8893, -0.9466],
#          [-1.2417,  1.6891],
#          [-0.7148,  0.1376],
#          [-0.2162, -0.6122],
#          [ 2.4048, -0.3211],
#          [-0.1516,  0.4997],
#          [ 1.5298, -0.2291],
#          [ 1.3895,  1.2602]]),
#  tensor([[ 2.6073],
#          [-0.5787],
#          [ 5.6339],
#          [-4.0211],
#          [ 2.3117],
#          [ 5.8492],
#          [10.0926],
#          [ 2.1932],
#          [ 8.0441],
#          [ 2.6943]])]
```

## 定义模型

当我们在《线性回归的从零开始实现》中实现线性回归时，
我们明确定义了模型参数变量，并编写了计算的代码，这样通过基本的线性代数运算得到输出。
但是，如果模型变得更加复杂，且当我们几乎每天都需要实现模型时，自然会想简化这个过程。
这种情况类似于为自己的博客从零开始编写网页。
做一两次是有益的，但如果每个新博客就需要工程师花一个月的时间重新开始编写网页，那并不高效。

对于标准深度学习模型，我们可以[**使用框架的预定义好的层**]。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。
我们首先定义一个模型变量`net`，它是一个`Sequential`类的实例。
`Sequential`类将多个层串联在一起。
当给定输入数据时，`Sequential`实例将数据传入到第一层，
然后将第一层的输出作为第二层的输入，以此类推。
在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。
但是由于以后几乎所有的模型都是多层的，在这里使用`Sequential`会让你熟悉“标准的流水线”。

回顾 :numref:`fig_single_neuron`中的单层网络架构，
这一单层被称为*全连接层*（fully-connected layer），
因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。

在PyTorch中，全连接层在`Linear`类中定义。
值得注意的是，我们将两个参数传递到`nn.Linear`中。
第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

```python
#@tab pytorch
# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

## **初始化模型参数**

在使用`net`之前，我们需要初始化模型参数。
如在线性回归模型中的权重和偏置。
深度学习框架通常有预定义的方法来初始化参数。
在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，
偏置参数将初始化为零。

正如我们在构造`nn.Linear`时指定输入和输出尺寸一样，
现在我们能直接访问参数以设定它们的初始值。
我们通过`net[0]`选择网络中的第一个图层，
然后使用`weight.data`和`bias.data`方法访问参数。
我们还可以使用替换方法`normal_`和`fill_`来重写参数值。

```python
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

## 定义损失函数

[**计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数**]。
默认情况下，它返回所有样本损失的平均值。


```python
#@tab pytorch
loss = nn.MSELoss()
```

## 定义优化算法

小批量随机梯度下降算法是一种优化神经网络的标准工具，
PyTorch在`optim`模块中实现了该算法的许多变种。
当我们(**实例化一个`SGD`实例**)时，我们要指定优化的参数
（可通过`net.parameters()`从我们的模型中获得）以及优化算法所需的超参数字典。
小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。


```python
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

## 训练

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。
我们不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。
当我们需要更复杂的模型时，高级API的优势将大大增加。
当我们有了所有的基本组件，[**训练过程代码与我们从零开始实现时所做的非常相似**]。

回顾一下：在每个迭代周期里，我们将完整遍历一次数据集（`train_data`），
不停地从中获取一个小批量的输入和相应的标签。
对于每一个小批量，我们会进行以下步骤:

* 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
* 通过进行反向传播来计算梯度。
* 通过调用优化器来更新模型参数。

为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程。


```python
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```


下面我们[**比较生成数据集的真实参数和通过有限数据训练获得的模型参数**]。
要访问参数，我们首先从`net`访问所需的层，然后读取该层的权重和偏置。
正如在从零开始实现中一样，我们估计得到的参数与生成数据的真实参数非常接近。

```python
#@tab pytorch
w = net[0].weight.data
print('w的估计误差：', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```



## 小结

* 我们可以使用PyTorch的高级API更简洁地实现模型。
* 在PyTorch中，`data`模块提供了数据处理工具，`nn`模块定义了大量的神经网络层和常见损失函数。
* 我们可以通过`_`结尾的方法将参数替换，从而初始化参数。


## 练习

1. 如果将小批量的总损失`nn.MSELoss(reduction='sum)`替换为小批量损失的平均值`nn.MSELoss()`，需要如何更改学习率？
2. 查看深度学习框架文档，它们提供了哪些损失函数和初始化方法？用Huber损失代替原损失，即
    $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \text{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \text{ 其它情况}\end{cases}$$
3. 如何访问线性回归的梯度？

## 答案

- 问题1

将损失函数的定义方式中的reduction字段改为 'sum'的话，会使得梯度值放大为原来的num_example倍

使得在不改变学习率的情况下 很容易出现在最优解附近震荡的情况 降低学习效果

```python
# 将损失函数的计算方式定义为整个损失的和
loss = nn.MSELoss(reduction='sum')
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  #指定需要优化的参数 还有学习率
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y) #这里的net返回输入x经过定义的网络所计算出的值
        trainer.zero_grad() #清除上一次的梯度值 避免叠加上下面的数据所带来的梯度
        l.backward() #损失函数进行反向传播 求参数的梯度 
        trainer.step()  #trainer步进 根据指定的优化算法进行参数的寻优
    l = loss(net(features), labels) #根据上面的参数优化结果计算参数对整个数据集的拟合状态 以loss进行反映
    print(f'epoch {epoch + 1}, loss {l:f}')

# 在上述情况下改变学习率
print("\nchange learn rate\n")
trainer = torch.optim.SGD(net.parameters(), lr=0.03/10)  #指定需要优化的参数 还有学习率
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  #这里的net返回输入x经过定义的网络所计算出的值
        trainer.zero_grad()  #清除上一次的梯度值 避免叠加上下面的数据所带来的梯度
        l.backward()  #损失函数进行反向传播 求参数的梯度
        trainer.step()  #trainer步进 根据指定的优化算法进行参数的寻优
    l = loss(net(features), labels)  #根据上面的参数优化结果计算参数对整个数据集的拟合状态 以loss进行反映
    print(f'epoch {epoch + 1}, loss {l:f}')
```

- 总结
  - 明显可见地选用`sum`而不是`mean`将很大程度影响模型的学习效果
  - 原因：若不将损失函数采用均值 将会使得参数的梯度过大
  - 梯度经过放大后 原有的学习率显得过大 使得其出现了振荡 即步长过长导致
  - 在最优解的附近震荡 而无法高效逼近最优点
  ```text
    以平均值作为损失函数
    epoch 1, loss 0.000176
    epoch 2, loss 0.000094
    epoch 3, loss 0.000093
   
    以和作为损失函数，不改变学习率
    epoch 1, loss 0.109876
    epoch 2, loss 0.096990
    epoch 3, loss 0.101151
    
    以和作为损失函数，改变学习率
    epoch 1, loss 0.093118
    epoch 2, loss 0.093618
    epoch 3, loss 0.093380
  ```
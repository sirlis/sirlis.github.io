---
title: PyTorch基础（随机数种子）
date: 2020-10-26 15:22:19 +0800
categories: [Tutorial，Coding]
tags: [python]
math: true
---

本文主要记录自己学习 PyTorch 过程中涉及的一些基础知识。

<!--more-->

---
- [1。随机数](#1随机数)
  - [1.1。随机数产生](#11随机数产生)
  - [1.2。随机数种子](#12随机数种子)
- [2。张量比较](#2张量比较)
  - [2.1。max](#21max)
- [3。参考文献](#3参考文献)

# 1。随机数

## 1.1。随机数产生

随机数广泛应用在科学研究，但是计算机无法产生真正的随机数，一般成为伪随机数。它的产生过程：给定一个随机种子（一般是一个正整数），根据随机算法和种子产生随机序列。

给定相同的随机种子，计算机产生的随机数**列**是一样的（这也许是伪随机的原因）。比如：

```python
import random
print(random.random()) # 0.6347616556381207
print(random.random()) # 0.17717483228053954
random.seed(1024)
print(random.random()) # 0.7970515714521261
print(random.random()) # 0.4834988702079559
random.seed(1024)
print(random.random()) # 0.7970515714521261
print(random.random()) # 0.4834988702079559
```

可以看到，在设置随机数种子后，产生随机数的过程可以完全重复，这种特性非常适合比如神经网络权值初始化的复现。

## 1.2。随机数种子

随机种子是针对随机方法而言的。常见的随机方法有生成随机数，以及其他的像随机排序之类的，后者本质上也是基于生成随机数来实现的。在深度学习中，比较常用的随机方法的应用有：网络的随机初始化，训练集的随机打乱等。

当用户未指定随机种子，系统默认随机生成，一般与系统当前时间有关。用户指定随机种子后，使用随机函数产生的随机数可以复现。种子确定后，每次使用随机函数相当于从随机序列去获取随机数，每次获取的随机数是不同的。

使用 PyTorch 复现效果时，总是无法做到完全的复现。同一份代码运行两次，有时结果差异很大。这是由于算法中的随机性导致的。要想每次获得的结果一致，必须固定住随机种子。首先，我们需要找到算法在哪里使用了随机性，再相应的固定住随机种子。

```python
import numpy as np
import random
import os
import torch

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 设置 cpu 的随机数种子
    torch.cuda.manual_seed(seed) # 对于单张显卡，设置 gpu 的随机数种子
    torch.cuda.manual_seed_all(seed) # 对于多张显卡，设置所有 gpu 的随机数种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()
```
> https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317

其中，`torch.backends.cudnn.benchmark` 设置为 `True`，可以大大提升卷积神经网络的运行速度。它的原理是将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。这会导致网络的训练存在一定的随机性，导致训练结果存在一些微小的不确定。因此设置为 `False` 可以固定住训练结果。

>高斯定理. https://www.zhihu.com/question/67209417/answer/418568879

`torch.backends.cudnn.deterministic` 是啥？训练模型个人的基本要求是 deterministic/reproducible，或者说是可重复性。也就是说在随机种子固定的情况下，每次训练出来的模型要一样。之前遇到了不可重复的情况。训练CNN的时候，发现每次跑出来小数点后几位会有不一样。epoch 越多，误差就越多，虽然结果大致上一样，但是强迫症真的不能忍。后来发现在 PyTorch 0.3.0 的时候已经修复了这个问题，可以用 `torch.backends.cudnn.deterministic = True` 这样调用的 CuDNN 的卷积操作就是每次一样的了。

# 2。张量比较

## 2.1。max

不指定维度时，返回一个张量，为输入数据中的最大值

```python
>>> a = torch.randn(1，3)
>>> a
    tensor([[ 0.6763， 0.7445，-2.2369]])
>>> torch.max(a)
    tensor(0.7445)
```

指定维度时，返回一个 tuple，包含沿着该维度的最大值和对应的序号。

```python
>>> a = torch.randn(4，4)
>>> a
tensor([[-1.2360，-0.2942，-0.1222， 0.8475],
        [ 1.1949，-1.1127，-2.2379，-0.6702],
        [ 1.5717，-0.9207， 0.1297，-1.8768],
        [-0.6172， 1.0036，-0.6060，-0.2432]])
>>> torch.max(a，dim=1)
torch.return_types.max(values=tensor([0.8475，1.1949，1.5717，1.0036])，indices=tensor([3，0，0，1]))
```
对于二维张量，`dim=0` 沿列求最大（跨行间比较），`dim=1` 沿行求最大（跨列间比较）。

对于三维张量，构成为 `(通道，行，列)`，那么`dim=0` 通道间比较求最大，`dim=1` 跨行间比较求最大，`dim=2` 跨列间比较求最大。

```python

import torch
 
a = torch.randn(2,3,4) #随机生成数组
max_0=torch.max(a,dim=0) #针对第1个元素“2”，对应的是通道
max_1=torch.max(a,dim=1) #针对第2个元素“3”，对应的是行
max_2=torch.max(a,dim=2) #针对第2个元素“4”，对应的是列
print("a:\n"，a)
print("************************************************")
print("max(a)_0:"，max_0)  #dim=0,通道间进行比较，所以返回每一张特征图，同一像素位置上的最大值
print("max(a)_1:"，max_1)  #dim=1，行与行之间进行比较，所以返回每一张特征图，每一列的最大值
print("max(a)_2:"，max_2)  #dim=2，列与列之间进行比较，所以返回每一张特征图，每一行的最大值
 
<<
a:
 tensor([[[ 0.5323， 1.5229，-0.6122， 0.6054],
         [ 1.2424，-1.6005， 0.0779， 0.9227],
         [-0.6340，-0.5770，-0.1672， 0.3598]],
 
        [[-0.3770，-0.4992， 1.8444，-1.1040],
         [ 1.2238， 0.7283，-1.6462， 0.0325],
         [-0.3555，-0.2599， 1.5741， 1.0683]]])
************************************************
max(a)_0: (tensor([[ 0.5323， 1.5229， 1.8444， 0.6054],
        [ 1.2424， 0.7283， 0.0779， 0.9227],
        [-0.3555，-0.2599， 1.5741， 1.0683]])，tensor([[ 0， 0， 1， 0],
        [ 0， 1， 0， 0],
        [ 1， 1， 1， 1]]))
max(a)_1: (tensor([[ 1.2424， 1.5229， 0.0779， 0.9227],
        [ 1.2238， 0.7283， 1.8444， 1.0683]])，tensor([[ 1， 0， 1， 1],
        [ 1， 1， 0， 2]]))
max(a)_2: (tensor([[ 1.2424， 1.5229， 0.0779， 0.9227],
        [ 1.2238， 0.7283， 1.8444， 1.0683]])，tensor([[ 1， 0， 1， 1],
```

也就是说，`dim` 参数是按照张量维度从左到右、从外到内的顺序进行比较的。

# 3。参考文献

[1] [梦并不遥远](https://www.cnblogs.com/zyg123/)。[4.3Python数据处理篇之Matplotlib系列(三)---plt.plot()](https://www.cnblogs.com/zyg123/p/10504633.html).

[2] [我的明天不是梦](https://www.cnblogs.com/xiaoboge/)。[python使用matplotlib:subplot绘制多个子图](https://www.cnblogs.com/xiaoboge/p/9683056.html).
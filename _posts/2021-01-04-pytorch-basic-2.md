---
title: PyTorch基础（随机数种子）
date: 2021-01-04 15:22:19 +0800
categories: [Tutorial，Coding]
tags: [python]
math: true
---

本文主要记录自己学习 PyTorch 过程中涉及的一些基础知识。

<!--more-->

---
- [1. 随机数](#1-随机数)
  - [1.1. 随机数产生](#11-随机数产生)
  - [1.2. 随机数种子](#12-随机数种子)
- [2. 参考文献](#2-参考文献)

# 1. 随机数

## 1.1. 随机数产生

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

## 1.2. 随机数种子

随机种子是针对随机方法而言的。常见的随机方法有生成随机数，以及其他的像随机排序之类的，后者本质上也是基于生成随机数来实现的。在深度学习中，比较常用的随机方法的应用有：网络的随机初始化，训练集的随机打乱等。

当用户未指定随机种子，系统默认随机生成，一般与系统当前时间有关。用户指定随机种子后，使用随机函数产生的随机数可以复现。种子确定后，每次使用随机函数相当于从随机序列去获取随机数，每次获取的随机数是不同的。

使用 **PyTorch** 复现效果时，总是无法做到完全的复现。同一份代码运行两次，有时结果差异很大。这是由于算法中的随机性导致的。要想每次获得的结果一致，必须固定住随机种子。首先，我们需要找到算法在哪里使用了随机性，再相应的固定住随机种子。

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

其中

- `torch.backends.cudnn.enabled`：cuDNN 使用非确定性算法。如果该参数设置为 `True`，说明设置为使用使用**非确定**性算法；

- `torch.backends.cudnn.benchmark`：在 `torch.backends.cudnn.enabled = True` 的前提下将该参数设置为 `True`，可以让程序在开始时花费一点额外时间，自动为整个网络的每个卷积层搜索最适合它的卷积实现算法，来达到优化运行效率的目的。但这会导致网络的训练存在一定的随机性，导致训练结果存在一些微小的不确定。一般来讲，应该遵循以下准则：

  - 如果网络的输入数据维度或类型上变化不大，设置 `torch.backends.cudnn.benchmark = True` 可以增加运行效率；
  - 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。

> https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317
> https://www.cnblogs.com/wanghui-garcia/p/11514502.html

- `torch.backends.cudnn.deterministic`：`torch.backends.cudnn.benchmark = True`会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。比如训练CNN的时候，发现每次跑出来小数点后几位会有不一样。epoch 越多，误差就越多，虽然结果大致上一样，但是强迫症真的不能忍。如果想要避免这种结果波动，可以设置 `torch.backends.cudnn.deterministic = True` 这样调用的 CuDNN 的卷积操作就是每次一样的了。

>高斯定理. https://www.zhihu.com/question/67209417/answer/418568879


# 2. 参考文献

[1] [梦并不遥远](https://www.cnblogs.com/zyg123/)。[4.3Python数据处理篇之Matplotlib系列(三)---plt.plot()](https://www.cnblogs.com/zyg123/p/10504633.html).

[2] [我的明天不是梦](https://www.cnblogs.com/xiaoboge/)。[python使用matplotlib:subplot绘制多个子图](https://www.cnblogs.com/xiaoboge/p/9683056.html).
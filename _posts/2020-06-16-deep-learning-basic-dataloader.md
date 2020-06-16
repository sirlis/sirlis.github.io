---
layout: post
title:  "深度学习基础（PyTorch的数据集）"
date:   2020-06-16 16:24:19
categories: Coding
tags: Python
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


# 目录

* [目录](#目录)
* [Torchvision](#Torchvision)
* [Dataset](#Dataset)
  * [默认类](#自定义类)
  * [自定义类](#自定义类)
* [DataLoader](#DataLoader)
* [参考文献](#参考文献)

# torchvision

`torchvision` 是 PyTorch 中专门用来处理图像的库，PyTorch 官网的安装教程也会让你安装上这个包。这个包中有四个大类。

- torchvision.datasets

- torchvision.models

- torchvision.transforms

- torchvision.utils

这里我们主要介绍前三个。

`torchvision.datasets` 是用来进行数据加载的，PyTorch团队在这个包中帮我们提前处理好了很多很多图片数据集。参考 [PyTorch中文文档](#https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-datasets/) 中的相关介绍。

- MNIST
- COCO
- Captions
- Detection
- LSUN
- ImageFolder
- Imagenet-12
- CIFAR
- STL10
- SVHN
- PhotoTour

我们可以直接使用这些数据集，示例如下：

```python
mnist_train_data = torchvision.datasets.MNIST('mnist/', train=True, download=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=16, shuffle=True)
```

`torchvision.datasets` 是 `torch.utils.data.Dataset` 的一个子类，所以他们也可以通过 `torch.utils.data.DataLoader` 使用多线程（python的多进程）。比如

```python
torch.utils.data.DataLoader(coco_cap, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)
```

# Dataset

## 默认类

`torch.utils.data.Dataset` 是一个抽象类，是 `Pytorch` 中图像数据集中最为重要的一个类，也是 `Pytorch` 中所有数据集加载类中应该继承的父类。用户想要加载**自定义的数据必须继承这个类**，并且覆写其中的两个方法：

- `__len__`：实现 `len(dataset)` 返回整个数据集的大小。

- `__getitem__`：用来获取一些索引的数据，使 `dataset[i]` 返回数据集中第 `i` 个样本。

不覆写这两个方法会直接返回错误。

这个类其实也就是起到了封装我们加载函数的作用，在继承了这个 `Dataset` 类之后，我们需要实现的核心功能便是 `__getitem__()`函数，`__getitem__()` 是 `Python` 中类的默认成员函数，我们通过实现这个成员函数实现可以通过索引来返回图像数据的功能。那么怎么得到图像从而去返回呢？当然不会直接将图像数据加载到内存中，相反我们只需要得到图像的地址就足够了，然后在调用的时候通过不同的读取方式读取即可。

不同的读取方式参见：[python深度学习库pytorch::transforms练习:opencv,scikit-image,PIL图像处理库比较](#https://oldpan.me/archives/pytorch-transforms-opencv-scikit-image)。

## 自定义类

更多的时候我们需要使用自己的数据集，数据集的形式可能为原始图片、可能为数组。下面以原始图片为例创建自己的数据集。要创建用于分类的自定义数据集，需要准备**两部分**内容：

- 图片数据集

- 标签信息（可用txt文件、csv文件记录，或通过图片文件名划分）

下面是自定义一个Dataset的代码示例

```python
class CustomDataset(torch.utils.data.Dataset):# Need to inherit `data.Dataset`
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0
```

按照上述模板定义一个自定义数据类，原始数据为 `.jpg` 图像。采用文件名来定义标签信息，图像文件的命名规范为：

```python
[label_no].[label_name].[image_num].jpg # '0.satellite.01.jpg' with label no '0' and label name 'satellite'
```

即文件名由 `.` 分隔，第一个数字为 `label` 的编号，第二个字符串为标签名称。采用 `scikit_image` 包读取图像，则自定义数据类如下

```python
import numpy as np
from skimage import io  # scikit-image
from torch.utils.data import Dataset
import os

def open_image(image_path):
    return io.imread(image_path)  # load image by scikit-image

'''
support jpg images only.
image file name should be: [label_no].[label_name].[image_num].jpg, 
thus label can be extracted from file name.
'''

class SPACE(Dataset):
    def __init__(self, root, train=True, augment=False, transform=None):
        self.train = train
        self.augment = augment
        self.transform = transform
        if self.train:
            self.data = np.array([
                x.path for x in os.scandir(root + "train\\")
                if x.name.endswith(".jpg") or x.name.endswith(".JPG")
            ])
            filename = np.array([
                os.path.split(x.path)[1] for x in os.scandir(root + "train\\")
                if x.name.endswith(".jpg") or x.name.endswith(".JPG")
            ])
            self.label = np.array([x.split('.', 1)[0] for x in filename])
        else:
            self.data = np.array([
                x.path for x in os.scandir(root + "test\\")
                if x.name.endswith(".jpg") or x.name.endswith(".JPG")
            ])
            filename = np.array([
                os.path.split(x.path)[1] for x in os.scandir(root + "test\\")
                if x.name.endswith(".jpg") or x.name.endswith(".JPG")
            ])
            self.label = np.array([x.split('.', 1)[0] for x in filename])

    def __getitem__(self, index):
        label = self.label[index]
        image = open_image(self.data[index])
        if self.augment:
            image = self.augment(image)  # augment images
        if self.transform is not None:
            image = self.transform(image)  # transform images
        return image, label

    def __len__(self):
        return len(self.data)  # return image number
```

在后续使用数据集时

```python
for batch_index, (data, target) in dataloader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
```


为什么直接能用 `for batch_index, (data, target) In dataloader` 这样的语句呢？

其实这个语句还可以这么写：

```python
for batch_index, batch in train_loader
    data, target = batch
```

这样就好理解了，因为这个迭代器每一次循环所得的batch里面装的东西，就是我在 `__getitem__` 方法最后 return 回来的，所以想在训练或者测试的时候还得到其他信息的话，就去增加一些返回值即可，只要是能return出来的，就能在每个batch中读取到。

# DataLoader

`torch.utils.data.DataLoader`  的核心参数包括[[1](#ref1)]：

- **`dataset`**：Dataset，输入数据集；

- `batch_size`：int，每批加载多少样本，default=1；

- `shuffle`：bool，是否打乱顺序，default=False；

- `sampler`：Sampler，定义从数据集中加载样本的策略，如果定义，则 ``shuffle`` 必须设为 ``False``；

- `num_workers`：int，采用多少个子进程加载数据集，0表示仅在主进程加载，default = 0；

- `pin_memory`：如果设为 ``True``，`DataLoader` 会将 tensors 会将拷贝到 CUDA 的锁页内存中，然后再返回它们，default = ``False``；
- `drop_last`：设为 ``True`` 扔掉最后一个不完整的 batch。如果数据集大小无法被 ``batch_size`` 整除，那么最后一披数据不完整，default = ``False``； 

**num_workers** 这个参数必须大于等于0，0的话表示数据导入在主进程中进行，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度。

**pin_memory** 就是锁页内存，创建DataLoader时，设置 ``pin_memory=True``，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。而显卡中的显存全部是锁页内存！当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

```python

```

# 参考文献

<span id="ref1">[1]</span>  [cdy艳0917](https://me.csdn.net/sinat_42239797). [Pytorch学习（三）定义自己的数据集及加载训练](https://blog.csdn.net/sinat_42239797/article/details/90641659?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase).

<span id="ref2">[2]</span>  teeyohuang. [Pytorch打怪路（三）Pytorch创建自己的数据集1](https://blog.csdn.net/Teeyohuang/article/details/79587125).

<span id="ref3">[3]</span>  Vincent Dumoulin, Francesco Visin. [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) ([Github](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)).

<span id="ref4">[4]</span> 知乎. [PyTorch 中，nn 与 nn.functional 有什么区别？](https://www.zhihu.com/question/66782101).

<span id="ref5">[5]</span>  PyTorch. [MaxPool2d](https://pytorch.org/docs/stable/nn.html#maxpool2d).

<span id="ref6">[6]</span>  PyTorch. [nn.Linear](https://pytorch.org/docs/stable/nn.html#linear).
---
layout: post
title:  "深度学习基础（PyTorch的DataLoader）"
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
* [层](#层)
  * [Conv2d](#Conv2d)
    * [dilation](#dilation)
    * [padding](#padding)
  * [MaxPool2d](#MaxPool2d)
  * [Linear](#Linear)
* [激活函数](#激活函数)
  * [conv2d](#conv2d)
  * [softmax](#softmax)
  * [log_softmax](#log_softmax)
* [损失函数](#损失函数)
  * [CrossEntropyLoss](#CrossEntropyLoss)
  * [NLLLoss](#NLLLoss)
* [参考文献](#参考文献)

# 层

## dataset



## DataLoader

`torch.utils.data.DataLoader`  的参数包括[[1](#ref1)]：

- **`in_channels`**：int，输入图片的通道数（彩色图像=3，灰度图像=1）；

- **`out_channels`**：int，卷积输出图片的通道数（也就是卷积核个数）；

- **`kernel_size`**：int或tuple，卷积核尺寸（赋值单个int时长=宽），default=1；

- `stride`：int或tuple，卷积操作的滑动步长，default=1；

- `padding`：int或tuple，输入图片外围扩充大小（赋值单个int时长=宽），default=0；


​       当采取默认参数时，padding = (kernel_size - 1) /2 可保证输出图片与输入图片尺寸一致；

- `dilation`：卷积核扩充大小，default=1；

- `groups`：从输入通道到输出通道分组的个数，default=1；

- `bias`：bool，输出增加偏差，default=True；

num_workers，从注释可以看出这个参数必须大于等于0，0的话表示数据导入在主进程中进行，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度。

pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。

主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。

而显卡中的显存全部是锁页内存！

当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

的输入为 `(batch_size, channel, height, width)`。

```python
class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).  每个 epoch 重新随机数据
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.  定义抽样方法
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: ``0``)  多少个线程 用于 加载数据
        collate_fn (callable, optional): merges a list of samples to form a mini-batch. 把 list sample 合并成 mini-batch
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your ``collate_fn`` returns a batch that is a custom type
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)  当 batch 很大是，最后一轮可能样本数量偏少，影响模型训练
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
    """
```

# 参考文献

<span id="ref1">[1]</span>  PyTorch. [Conv2d](https://pytorch.org/docs/stable/nn.html#conv2d).

<span id="ref2">[2]</span>  Stack Overflow. [Default dilation value in PyTorch](https://stackoverflow.com/questions/43474072/default-dilation-value-in-pytorch).

<span id="ref3">[3]</span>  Vincent Dumoulin, Francesco Visin. [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) ([Github](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)).

<span id="ref4">[4]</span> 知乎. [PyTorch 中，nn 与 nn.functional 有什么区别？](https://www.zhihu.com/question/66782101).

<span id="ref5">[5]</span>  PyTorch. [MaxPool2d](https://pytorch.org/docs/stable/nn.html#maxpool2d).

<span id="ref6">[6]</span>  PyTorch. [nn.Linear](https://pytorch.org/docs/stable/nn.html#linear).
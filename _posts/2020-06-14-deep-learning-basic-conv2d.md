---
layout: post
title:  "深度学习基础（2D卷积层）"
date:   2020-06-13 16:24:19
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
* [nn.Conv2d](#nn.Conv2d)
  * [epoch](#epoch)
  * [batch & batch_size](#batch & batch_size)
  * [iteration](#iteration)
* [优化器](#优化器)
  * [批量梯度下降（BGD）](#批量梯度下降（BGD）)
  * [随机梯度下降（SGD）](#随机梯度下降（SGD）)
  * [小批量梯度下降（MBGD）](#小批量梯度下降（MBGD）)
  * [梯度下降的缺点](#梯度下降的缺点)
  * [动量（Momentum）](#动量（Momentum）)
  * [RMSProp](#RMSProp)
  * [Adam](#Adam)
* [参考文献](#参考文献)

# nn.Conv2d

`nn.Conv2d` 的参数包括[[1](#ref1)]：

- **`in_channels`**：int，输入图片的通道数（彩色图像=3，灰度图像=1）；
- **`out_channels`**：int，卷积输出图片的通道数（也就是卷积核个数）；
- **`kernel_size`**：int或tuple，卷积核尺寸（赋值单个int时长=宽），default=1；
- `stride`：int或tuple，卷积操作的步长，default=1；
- `padding`：int或tuple，输入图片外围扩充大小（赋值单个int时长=宽），default=0；

​                           当采取默认参数时，padding = (kernel_size - 1) /2 可保证输出图片与输入图片尺寸一致；

- `dilation`：卷积核扩充大小，default=1；
- `groups`：从输入通道到输出通道分组的个数，default=1；
- `bias`：bool，输出增加偏差，default=True；

假设输入图片的高度和宽度为 $H_{in}$ 和 $H_{in}$，则输出特征图的高度和宽度$H_{out}$ 和 $H_{out}$ 为

$$
H_{out} = \frac{H_{in} + 2\times padding[0] - dilation[0]\times (kernel\_size[0]-1)-1} {stride[0]}+1 \\
W_{out} = \frac{W_{in} + 2\times padding[1] - dilation[1]\times (kernel\_size[1]-1)-1}{stride[1]}+1
$$

为了保证输出尺寸与输入尺寸一致，即
$$
H_{out} = H_{in} \\
W_{out} = W_{in}
$$
需要计算匹配的 `padding` 值，将上面的公式变换如下

$$
padding[0] = \frac{stride[0]\times (H_{out}-1) + dilation[0]\times (kernel\_size[0]-1)-H_{in}+1} {2} \\
padding[1] = \frac{stride[1]\times (H_{out}-1) + dilation[1]\times (kernel\_size[1]-1)-H_{in}+1} {2} \\
$$

将默认的 `stride=1`和 `dilation=1` 参数代入，可得

$$
padding[0] = (kernel\_size[0]-1) / 2 \\
padding[1] = (kernel\_size[1]-1) / 2
$$

也就是说，`padding` 的取值与 `kernel_size` 有关。如果采用长宽相等的卷积核，可简写为

$$
padding = (kernel\_size-1) / 2 \\
$$

## dilation

**注意**，PyTorch 认为`dilation=n` 表示卷积核尺寸从 (`1x1`) 扩充为 `nxn`，其中原本的卷积核像素在左上角，其它像素填充为0。因此 `dilation=1` 等价于传统的无扩充的卷积[[2](#ref2)]。

如果我们设置的 `padding=0, dilation=1` 的话，蓝色为输入，绿色为输出，卷积核为3*3的卷积效果如图[[3](#ref3)]：

![01.dilation](..\assets\img\postsimg\20200614\01.no_padding_no_strides.gif)

如果我们设置的 `dilation=2` 的话，卷积核点与输入之间距离为1的值相乘来得到输出  

![02.dilation1](..\assets\img\postsimg\20200614\02.dilation1.png)

这样单次计算时覆盖的面积（即感受域）由 `dilation=0` 时的 $3\times 3=9$ 变为了 `dilation=1` 时的 $5\times 5=25$。在增加了感受域的同时却没有增加计算量，保留了更多的细节信息，对图像还原的精度有明显的提升。

# nn.MaxPool2d

`nn.MaxPool2d` 的参数包括：

- `kernel_size`：(int or tuple) ，max pooling 的窗口大小，可以为tuple，在nlp中tuple用更多（n,1）
- `stride`：(int or tuple, optional) ，max pooling 的窗口移动的步长。默认值是kernel_size
- `padding`：(int or tuple, optional) - 输入的每一条边补充0的层数
- `dilation`：(int or tuple, optional) – 一个控制窗口中元素步幅的参数
- `return_indices`：如果等于 `True`，会返回输出最大值的序号，对于上采样操作会有帮助
- `ceil_mode`：如果等于 `True`，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

## epoch

一个 **epoch** 指将训练集中**全部样本**送入神经网络完成一次训练（前向计算及反向传播）的过程。在实际训练时，将所有训练集数据迭代训练一次是不够的，需要反复多次才能拟合收敛。因此，往往设置多次epoch进行训练。

随着epoch数量的增加，神经网络中权重更新迭代的次数增多，曲线从最开始的不拟合状态，慢慢进入优化拟合状态，最终进入过拟合。因此，epoch的个数是非常重要的。那么究竟设置为多少才合适呢？恐怕没有一个确切的答案。对于不同的数据库来说，epoch数量是不同的。但是，epoch大小与数据集的多样化程度有关，多样化程度越强，epoch应该越大。

## batch & batch_size

由于训练集中所有样本的数据往往很大，如果一次就将训练集中的所有样本数据送入计算机，计算机将无法负荷。因此，一般会将整个训练集数据分成几个较小的批（**batch**es），每批采用一定数量的样本进行训练。**batch_size** 即为每批训练使用的样本的个数。

## iteration

一个 iteration 指采用 batch_size 个样本训练一次的过程。

# 优化器

## 批量梯度下降（BGD）

最早期的机器学习采用批量梯度下降（Batch Gradient Decent，BGD）进行参数更新[[2](#ref2)]。梯度下降法在每个 epoch 中需要对**全部**样本进行梯度计算，然后取平均值进行权值更新。即 batch_size 等于训练集样本个数。当数据集较小时该方法尚可。随着数据集迅速增大，这种方法一次开销大进而占用内存过大，速度过慢。

Batch gradient descent 对于凸函数可以收敛到全局极小值，对于非凸函数可以收敛到局部极小值。

## 随机梯度下降（SGD）

后来产生了一次只训练一个样本的方法（batchsize=1），称为随机梯度下降（Stochastic Gradient Decent，SGD）。该方法根据每次只使用**一个**样本的情况更新一次权值，开销小速度快，但由于单个样本的巨大随机性，全局来看优化性能较差，收敛速度很慢，产生局部震荡，有限迭代次数内很可能无法收敛。

 SGD 一次只进行一次更新，就没有冗余，而且比较快，并且可以新增样本。缺点是SGD的噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。所以虽然训练速度快，但是准确度下降，并不是全局最优。虽然包含一定的随机性，但是从期望上来看，它是等于正确的导数的。

## 小批量梯度下降（MBGD）

后来诞生的小批量梯度下降（Mini-Batch Gradient Descent ，MBGD），相当于上述两个“极端”方法的折中：将训练集分成多个mini_batch（即常说的**batch**）,一次迭代训练一个minibatch（即**batch_size**个样本），根据该batch数据的loss更新权值。

MBGD 每一次利用一小批样本，即 batch_size 个样本进行计算，这样它可以降低参数更新时的方差，收敛更稳定，另一方面可以充分地利用深度学习库中高度优化的矩阵操作来进行更有效的梯度计算。

## 梯度下降的缺点

两大缺点[[3](#ref3)]：

- **MBGD不能保证很好的收敛性**，learning rate 如果选择的太小，收敛速度会很慢，如果太大，loss function 就会在极小值处不停地震荡甚至偏离。（有一种措施是先设定大一点的学习率，当两次迭代之间的变化低于某个阈值后，就减小 learning rate，不过这个阈值的设定需要提前写好，这样的话就不能够适应数据集的特点。）**对于非凸函数，还要避免陷于局部极小值处，或者鞍点处，因为鞍点周围的error是一样的，所有维度的梯度都接近于0，SGD 很容易被困在这里。（**会在鞍点或者局部最小点震荡跳动，因为在此点处，如果是训练集全集带入即 BGD，则优化会停止不动，如果是 mini-batch 或者 SGD，每次找到的梯度都是不同的，就会发生震荡，来回跳动。）
- **SGD对所有参数更新时应用同样的 learning rate**，如果我们的数据是稀疏的，我们更希望对出现频率低的特征进行大一点的更新。learning rate 会随着更新的次数逐渐变小。

## 动量（Momentum）

为了应对第一个缺点，采用动量优化器，**可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。**

$$
v_{d\omega}=\beta v_{d\omega}+(1−\beta)d\omega \\
v_{db}=\beta v_{db}+(1−\beta)db \\
\omega=\omega−\eta v_{d\omega} \\
b=b−\eta v_{db} \\
$$

其中，在上面的公式中$v_{d\omega}$和$v_{db}$分别是损失函数在前 $t−1$ 轮迭代过程中累积的梯度梯度动量，$\beta$ 是梯度累积的一个指数，这里我们一般设置值为0.9。所以Momentum优化器的主要思想就是利用了类似与移动指数加权平均的方法来对网络的参数进行平滑处理的，让梯度的摆动幅度变得更小。$d\omega$和$db$分别是损失函数反向传播时候所求得的梯度，下面两个公式是网络权重向量和偏置向量的更新公式，$\eta$ 是网络的学习率。

当我们使用 Momentum 优化算法的时候，可以解决 mini-batch SGD 优化算法更新幅度摆动大的问题，同时可以使得网络的收敛速度更快。

## RMSProp

RMSProp算法的全称叫 Root Mean Square Prop，是Geoffrey E. Hinton在Coursera课程中提出的一种优化算法，是对Momentum优化算法的进一步优化。为了进一步优化损失函数在更新中存在摆动幅度过大的问题，并且进一步加快函数的收敛速度，RMSProp算法对权重 $W$ 和偏置 $b$ 的梯度使用了指数加权平均数。

其中，假设在第 $t$ 轮迭代过程中，各个公式如下所示

$$
s_{d\omega}=\beta s_{d\omega}+(1−\beta)d\omega^2 \\
s_{d\omega}=\beta s_{db}+(1−\beta)db^2 \\
\omega=\omega−\eta \frac{d\omega}{\sqrt{s_{d\omega}}+\epsilon} \\
b=b−\eta \frac{db}{\sqrt{s_{db}}+\epsilon} \\
$$

这个分母相当于梯度的**均方根 （Root Mean Squared，RMS）**。RMSProp算法对梯度计算了微分平方加权平均数。这种做法有利于消除了摆动幅度大的方向，用来修正摆动幅度，使得各个维度的摆动幅度都较小。另一方面也使得网络函数收敛更快。（比如当 $d\omega$ 或者 $db$ 中有一个值比较大的时候，那么我们在更新权重或者偏置的时候除以它之前累积的梯度的平方根，这样就可以使得更新幅度变小）。为了防止分母为零，使用了一个很小的数值 $\epsilon$ 来进行平滑，一般取值为$10^{−8}$。

## Adam

有了上面两种优化算法，一种可以使用类似于物理中的动量来累积梯度，另一种可以使得收敛速度更快同时使得波动的幅度更小。那么讲两种算法结合起来所取得的表现一定会更好。Adam（Adaptive Moment Estimation）算法是将Momentum算法和RMSProp算法结合起来使用的一种算法，我们所使用的参数基本和上面讲的一致，在训练的最开始我们需要初始化梯度的累积量和平方累积量[[4](#ref4)]。

假设在训练的第 $t$ 轮训练中，我们首先可以计算得到Momentum和RMSProp的参数更新

$$
v_{d\omega}=\beta_1 v_{d\omega}+(1−\beta_1)d\omega \\
v_{db}=\beta_1 v_{db}+(1−\beta_1)db \\
s_{d\omega}=\beta s_{d\omega}+(1−\beta)d\omega^2 \\
s_{d\omega}=\beta s_{db}+(1−\beta)db^2 \\
$$

参数 $\beta_1$ 所对应的就是Momentum算法中的 $\beta$ 值，一般取0.9；参数 $\beta_2$ 所对应的就是RMSProp算法中的 $\beta$ 值，一般取0.999。

由于移动指数平均在迭代开始的初期会导致和开始的值有较大的差异，所以我们需要对上面求得的几个值做偏差修正

$$
v^c_{d\omega}=\frac{v_{d\omega}}{1−\beta^t_1} \\
v^c_{db}=\frac{v_{d b}}{1−\beta^t_1} \\
s^c_{d\omega}=\frac{s_{d\omega}}{1−\beta^t_2} \\
s^c_{db}=\frac{s_{db}}{1−\beta^t_2} \\
$$

通过上面的公式，我们就可以求得在第 $t$ 轮迭代过程中，参数梯度累积量的修正值，从而接下来就可以根据Momentum和RMSProp算法的结合来对权重和偏置进行更新

$$
\omega=\omega−\eta \frac{v^c_{d\omega}}{\sqrt{s_{d\omega}}+\epsilon} \\
b=b−\eta \frac{v^c_{db}}{\sqrt{s_{db}}+\epsilon} \\
$$

其中 $\epsilon$ 是一个平滑项，我们一般取值为 $10^{−8}$，学习率 $\eta$ 则需要我们在训练的时候进行微调。

# 参考文献

<span id="ref1">[1]</span>  PyTorch. [Conv2d](https://pytorch.org/docs/stable/nn.html#conv2d).

<span id="ref2">[2]</span>  Stack Overflow. [Default dilation value in PyTorch](https://stackoverflow.com/questions/43474072/default-dilation-value-in-pytorch).

<span id="ref3">[3]</span>  Vincent Dumoulin, Francesco Visin. [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) ([Github](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)).

<span id="ref2">[2]</span> [LLLiuye](https://www.cnblogs.com/lliuye/). [批量梯度下降(BGD)、随机梯度下降(SGD)以及小批量梯度下降(MBGD)的理解](https://www.cnblogs.com/lliuye/p/9451903.html).

<span id="ref3">[3]</span> [郭耀华](https://home.cnblogs.com/u/guoyaohua/). [深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）](https://www.cnblogs.com/guoyaohua/p/8542554.html).

<span id="ref4">[4]</span> [William](https://me.csdn.net/willduan1). [深度学习优化算法解析(Momentum, RMSProp, Adam)）](https://blog.csdn.net/willduan1/article/details/78070086).
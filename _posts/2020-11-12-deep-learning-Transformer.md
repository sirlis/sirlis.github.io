---
title: 深度学习文章阅读（Transformer）
date: 2020-11-12 17:04:19 +0800
categories: [Knowledge, DeepLearning]
tags: [academic]
math: true
---

本文主要介绍 seq2seq learning 中的 Transformer 模型，由谷歌提出。

<!--more-->

---
- [1. 简介](#1-简介)
- [2. 总体结构](#2-总体结构)
  - [2.1. Encoder](#21-encoder)
    - [2.1.1. input](#211-input)
    - [2.1.2. positional encoding](#212-positional-encoding)
- [3. 参考文献](#3-参考文献)


# 1. 简介

Transformer 来自 Google 团队 2017 年的文章 **《Attenion Is All You Need》**（https://arxiv.org/abs/1706.03762 ），该文章的目的：减少计算量并且提高并行效率，同时不减弱最终的实验效果。Transformer 在机器翻译任务上的表现超过了 RNN、CNN，只用 encoder-decoder 和 attention 机制就能达到很好的效果，最大的优点是可以高效地并行化。

自 attention 机制提出后，加入 attention 的 seq2seq 模型在各个任务上都有了提升，所以现在的 seq2seq 模型指的都是结合 RNN 和 attention 的模型。之后 google 又提出了解决 seq2seq 问题的 Transformer 模型，用全 attention 的结构代替了 lstm，在翻译任务上取得了更好的成绩。

# 2. 总体结构

模型结构如下图所示

![attention](../assets/img/postsimg/20201112/1.jpg)

和大多数 seq2seq 模型一样，transformer 的结构也是由 encoder 和 decoder 组成。Transformer 的 encoder 由 6 个编码器叠加组成，decoder 也由 6 个解码器组成，在结构上都是相同的，但它们不共享权重。图中左边的部分就是 Encoder，由 6 个相同的 layer 组成，layer 指的就是上图左侧的单元，最左边有个 “Nx”，这里是 $x=6$ 个。类似的，途中右边的部分就是 Decoder，同样由 6 个相同的 layer 组成。

从**顶层**看，Transformer 就是一个 Encoder-Decoder 框架的一种实现。

![attention](../assets/img/postsimg/20201112/1.1.jpg)

在编码端和解码端，分别堆叠了 6 个编码器 / 解码器。6 这个数字并没由什么特别理由，也可以换成其它数字。编码器和解码器的内部结构大同小异，都包含一个 Self-Attention 模块和一个 Feed Forward 模块，不同的是解码器部分中间还增加了一个 Encoder-Decoder Attention 模块。

## 2.1. Encoder

下面将目光聚焦到 Encoder，它由两个 sub-layer 组成，分别是

- multi-head **self-attention** mechanism
- fully connected **feed-forward** network

![attention](../assets/img/postsimg/20201112/1.2.jpg)

Encoder 的数据流通过程如下

- Input 经过 embedding 后，要做 positional encoding
- 然后是 Multi-head attention
- 再经过 position-wise Feed Forward
- 每个子层之间有残差连接

![attention](../assets/img/postsimg/20201112/2.jpg)

### 2.1.1. input

首先使用嵌入算法将输入的 word（$x$） 转换为 vector（$z$），这个转换仅在最下方第一个 Encoder 之前发生。在 NLP 任务中，假设每个单词都转化为 $d_{model}=512$ 维的向量，用下图中的 4 个框并排在一起表示。

![input](../assets/img/postsimg/20201112/3.jpg)

对于其它 Encoder 而言，同样是输入 512 维的向量，只不过第一个 Encoder 输入的是词嵌入向量，而其它 Encoder 输入其下方 Encoder 的输出向量。包含各个词向量的**列表长度**是一个超参数，一般设为训练数据集中最长句子的长度。

### 2.1.2. positional encoding

在数据预处理的部分，由于 Transformer 抛弃了卷积（convolution）和循环（recurrence），为了使得模型具备利用句子序列顺序的能力，必须要在词向量中插入一些相对或绝对位置信息。

Positional Encoding 是一种考虑输入序列中单词顺序的方法。Encoder 为每个输入词向量添加了一个维度（$d_{model}=512$）与词向量一致的位置向量 $PE$，这些位置向量符合一种特定模式，可以用来确定每个单词的位置，或者用来提供信息以衡量序列中不同单词之间的距离。

作者提出两种 Positional Encoding 的方法，将 encoding 后的数据与 embedding 数据求和，加入了相对位置信息。

- 固定方法：用不同频率的 $sine$ 和 $cosine$ 函数直接计算
- 学习方法：学习出一份 positional embedding

经过实验（[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)）发现两者的结果一样，所以最后选择了第一种方法。

$$
\begin{aligned}
PE_{(pos,2i)} &= sin(pos / 10000^{2i/d_{model}})\\
PE_{(pos,2i+1)} &= cos(pos / 10000^{2i/d_{model}})
\end{aligned}
$$

其中， $pos$ 是词在句子中的位置；$i$ 是位置向量的维度。每个位置向量的分量对应一个正弦或余弦函数。

下面详细分析一下这个位置向量的形式。从维度的角度来看，$i=0$ 时第一个维度由波长为 $2\pi$ 的正余弦函数构成。依次往后，第 $i$ 个维度对应的正余弦函数的波长逐渐变长（$10000^{2i/d_{model}}$）。最终波长从 $2\pi$ 到 $10000\cdot 2\pi$。作者选择正余弦函数的原因，是因为作者认为正余弦函数能够让模型轻松学习相对位置的参与，因为对于任何固定的偏移量 $k$，位置向量 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数

$$
\begin{aligned}
sin(PE_{pos+k}) &= sin(PE_{pos})cos(PE_k)+cos(PE_{pos})sin(PE_k)\\
cos(PE_{pos+k}) &= cos(PE_{pos})cos(PE_k)-sin(PE_{pos})sin(PE_k)\\
\end{aligned}
$$

这种方法相比学习而言还有一个好处，如果是学习到的 positional embedding，（个人认为，没看论文）会像词向量一样受限于词典大小。也就是只能学习到“位置2对应的向量是 (1,1,1,2) ” 这样的表示。而用三角公式明显不受序列长度的限制，也就是可以应对比训练时所用到序列的更长的序列。

将 positional embedding 可视化后的图如下所示

![pe](../assets/img/postsimg/20201112/5.jpg)

最后，将 $PE+wordvec$ 作为输入。如下图所示，假设 $wordvec$ 的维度为四个格子，那么实际的 positional encoding 如下所示

![position encoding](../assets/img/postsimg/20201112/4.jpg)


# 3. 参考文献

[1] Jay Alammar. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[1] 不会停的蜗牛. [图解什么是 Transformer](https://www.jianshu.com/p/e7d8caa13b21)

[2] rumor. [【NLP】Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)

[3] \_zhang_bei\_. [自然语言处理中的Transformer和BERT](https://blog.csdn.net/Zhangbei_/article/details/85036948)

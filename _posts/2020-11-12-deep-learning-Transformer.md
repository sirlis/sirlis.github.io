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
    - [2.1.1. positional encoding](#211-positional-encoding)
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

在编码端和解码端，分别堆叠了 6 个编码器 / 解码器。6 这个数字并没由什么特别理由，也可以换成其它数字。

## 2.1. Encoder

下面将目光聚焦到 Encoder，它由两个 sub-layer 组成，分别是

- multi-head self-attention mechanism
- fully connected feed-forward network

![attention](../assets/img/postsimg/20201112/1.2.jpg)

- Input 经过 embedding 后，要做 positional encoding
- 然后是 Multi-head attention
- 再经过 position-wise Feed Forward
- 每个子层之间有残差连接

![attention](../assets/img/postsimg/20201112/2.jpg)

- 首先使用嵌入算法将输入的 word（$x$） 转换为 vector（$z$）
- 下面的 sub-layer 输入是 embedding 向量
- 在 sub-layer 内部，输入向量经过 self-attention，再经过 feed-forward 层
- 该 sub-layer 的输出向量 $r$ 是它正上方 sub-layer 的输入
- 向量 $r$ 的大小是一个超参数，通常设置为训练集中最长句子的长度。

### 2.1.1. positional encoding

Positional Encoding 是一种考虑输入序列中单词顺序的方法。

encoder 为每个输入 embedding 添加了一个向量，这些向量符合一种特定模式，可以确定每个单词的位置，或者序列中不同单词之间的距离。

例如，input embedding 的维度为4，那么实际的positional encodings如下所示：

# 3. 参考文献

[1] Jay Alammar. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[1] 不会停的蜗牛. [图解什么是 Transformer](https://www.jianshu.com/p/e7d8caa13b21)

[2] rumor. [【NLP】Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)

[3] \_zhang_bei\_. [自然语言处理中的Transformer和BERT](https://blog.csdn.net/Zhangbei_/article/details/85036948)

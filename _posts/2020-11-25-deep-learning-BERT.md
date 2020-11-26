---
title: 深度学习文章阅读（BERT）
date: 2020-11-25 16:07:19 +0800
categories: [Knowledge, DeepLearning]
tags: [academic]
math: true
---

本文介绍了谷歌提出的 BERT 框架，基于 Transformer，在 NLP 领域的 11 个方向大幅刷新了精度，是近年来自残差网络最有突破性的一项技术。

<!--more-->

---
- [1. 简介](#1-简介)
- [2. 词向量模型](#2-词向量模型)
  - [word2vec](#word2vec)
  - [ELMo](#elmo)
- [3. 总体结构](#3-总体结构)
- [6. 参考文献](#6-参考文献)


# 1. 简介

> Devlin J, Chang M W, Lee K, et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**[J]. arXiv preprint arXiv:1810.04805v2, 2018.

BERT（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers）近期提出之后，作为一个 word2vec 的替代者，其在 NLP 领域的 11 个方向大幅刷新了精度，可以说是近年来自残差网络最优突破性的一项技术了。论文的主要特点以下几点：

- 使用了 Transformer 作为算法的主要框架，Transformer 能更彻底的捕捉语句中的双向关系；
- 使用了Mask Language Model(MLM) （Wilson L Taylor. 1953. cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30(4):415–433.）和 Next Sentence Prediction(NSP) 的多任务训练目标；
- 使用更强大的机器训练更大规模的数据，使 BERT 的结果达到了全新的高度，并且 Google 开源了 BERT 模型，用户可以直接使用 BERT 作为 Word2Vec 的转换矩阵并高效的将其应用到自己的任务中。

**BERT 的本质上是通过在海量的语料的基础上运行自监督学习方法为单词学习一个好的特征表示，即 BERT 是一个<font color=red>词向量模型</font>**。所谓自监督学习是指在没有人工标注的数据上运行的监督学习。在以后特定的 NLP 任务中，我们可以直接使用 BERT 的特征表示作为该任务的词嵌入特征。**所以 BERT 提供的是一个供其它任务迁移学习的模型，该模型可以根据任务微调或者固定之后作为特征提取器**。BERT 的源码和模型2019年10月31号已经在 Github 上开源，简体中文和多语言模型也于11月3号开源。

# 2. 词向量模型

这里主要横向比较一下 word2vec，ELMo，BERT 这三个模型，着眼在模型亮点与差别处。

传统意义上来讲，词向量模型是一个工具，可以把真实世界抽象存在的文字转换成可以进行数学公式操作的向量，而对这些向量的操作，才是 NLP 真正要做的任务。因而某种意义上，NLP 任务分成两部分，预训练产生词向量，对词向量操作（下游具体NLP任务）。

从 word2vec 到 ELMo 到 BERT，做的其实主要是把下游具体 NLP 任务的活逐渐移到预训练产生词向量上。下面是一个大体概括，具体解释后面会写到。。

- word2vec $\rightarrow$ ELMo：

结果：上下文无关的 static 向量变成上下文相关的 dynamic 向量，比如苹果在不同语境 vector 不同。

操作：encoder 操作转移到预训练产生词向量过程实现。

- ELMo $\rightarrow$ BERT：

结果：训练出的 word-level 向量变成 sentence-level 的向量，下游具体 NLP 任务调用更方便，修正了 ELMo 模型的潜在问题。

操作：使用句子级负采样获得句子表示/句对关系，Transformer 模型代替 LSTM 提升表达和时间上的效率，masked LM 解决 “自己看到自己” 的问题。

## word2vec

word2vec 模型其实就是简单化的神经网络。

## ELMo



# 3. 总体结构

BERT的网络架构使用的是《Attention is all you need》中提出的多层 Transformer 结构，其最大的特点是抛弃了传统的 RNN 和 CNN，通过 Attention 机制将任意位置的两个单词的距离转换成 1，有效的解决了 NLP 中棘手的长期依赖问题。详细可参考[此处](./deep-learning-Transformer/)。


# 6. 参考文献

[1] 大师兄. [BERT详解](https://zhuanlan.zhihu.com/p/48612853)

[1] 不会停的蜗牛. [图解什么是 Transformer](https://www.jianshu.com/p/e7d8caa13b21)

[2] rumor. [【NLP】Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)

[3] \_zhang_bei\_. [自然语言处理中的Transformer和BERT](https://blog.csdn.net/Zhangbei_/article/details/85036948)

[4] Amirhossein Kazemnejad. [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

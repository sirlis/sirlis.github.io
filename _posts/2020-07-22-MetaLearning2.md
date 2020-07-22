---
layout: post
title:  "元学习文章阅读（Prototypical Network）"
date:   2020-07-13 14:35:19
categories: Reading
tags: ML

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
* [Prototypical Network](#Prototypical Network)
  * [模型](#模型)
  * [关于二重梯度的进一步解释](#关于二重梯度的进一步解释)
  * [FOMAML一阶近似简化](#FOMAML一阶近似简化)
  * [缺点](#缺点)
* [Reptile](#Reptile)
  * [算法](#算法)
  * [分析](#分析)
  * [实验](#实验)
* [各类算法实现](#各类算法实现)
* [参考文献](#参考文献)

# Prototypical Network

2017.《Prototypical Networks for Few-shot Learning》

本文是2017年NIPS的会议论文，作者来自多伦多大学以及Twitter公司。在论文中作者提出了一种新的基于度量（Metric-based）的少样本学习模型——**原型网络（Prototypical Networks）**。原型网络首先利用 support  set 中每个类别提供的少量样本，计算它们的嵌入的中心，作为每一类样本的**原型（Prototype）**，接着基于这些原型学习一个度量空间，使得新的样本通过计算自身嵌入与这些原型的距离实现最终的分类，思想与聚类算法十分接近，但出发点有着很大的差异。除此之外，作者在文章中还尝试将原型网络应用于**零样本学习（Zero-shot learning）**问题上，通过数据集携带的属性向量形成**元数据（meta-data）**，基于这些元数据构建原型，从而实现零样本分类。

原型网络在少样本分类与零样本分类任务上的示意图如下所示。

![](..\assets\img\postsimg\20200722\1.jpg)

## 模型

在 few-shot 分类任务中，假设有 $N$ 个标记的样本 $S=\{(x_1,y_1),...,(x_N,y_N)\}$ ，其中 $x_i \in \mathbb R^D$ 是 $D$ 维的样本特征向量，$y \in \{1,...,K\}$ 是相应的 label 。$S_K$ 表示第 $k$ 类样本的集合。

原型网络计算每个类的 $M$ 维原型向量 $c_k \in \mathbb R^M$ ，计算的函数为 $f_{\phi}: \mathbb R^D \rightarrow \mathbb R^M$ ，其中 $\phi$ 为可学习参数。原型向量 $c_k$ 即为 embedding space 中该类的所有 support set 样本点的均值向量
$$
c_k = \frac{1}{|S_K|} \sum_{(x_i,y_i) \in S_K} f_{\phi}(x_i)
$$
给定一个距离函数 $d: \mathbb R^M \times \mathbb R^M \rightarrow [0,+\infin)$ ，原型网络通过在 embedding space 中对距离进行 **softmax** 计算，可以得到一个针对 $x$ 的样本点的概率分布
$$
p_{\phi}(y=k|x)=\frac{exp(-d(f_{\phi},c_k))}{\sum_{i}exp(-d(f_{\phi}(x),c_{i}))}
$$
通过在 SGD 中最小化第 $k$ 类的负对数似然函数 $J(\phi)$ 来推进学习
$$
J(\phi) = -log\; p(\phi)(y=k|x)
$$

## 算法

![image-20200722221410952](..\assets\img\postsimg\20200722\2.jpg)

其中

- $N$ 是训练集中的样本个数；
- $K$ 是训练集中的类个数；
- $N_C \leq K$ 是每个 episode 选出的类个数；
- $N_S$ 是每类中 support set 的样本个数；
- $N_Q$ 是每类中 query set 的样本个数；
- $RANDOMSAMPLE(S,N)$ 表示从集合 S 中随机选出 N 个元素。

下面详细分析算法。

首先，输入训练集

# 参考文献

<span id="ref1">[1]</span>  [Rust-in](https://www.zhihu.com/people/rustinnnnn). [MAML 论文及代码阅读笔记](https://zhuanlan.zhihu.com/p/66926599).

<span id="ref2">[2]</span> 人工智障. [MAML算法，model-agnostic metalearnings?](https://www.zhihu.com/question/266497742/answer/550695031)

<span id="ref3">[3]</span> [Veagau](https://www.cnblogs.com/veagau/). [【笔记】Reptile-一阶元学习算法](https://www.cnblogs.com/veagau/p/11816163.html)

[4] [pure water](https://blog.csdn.net/qq_41694504). [Reptile原理以及代码详解](https://blog.csdn.net/qq_41694504/article/details/106750606)
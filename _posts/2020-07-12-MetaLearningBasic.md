---
layout: post
title:  "元学习基础"
date:   2020-07-12 14:35:19
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
* [小样本学习](#小样本学习)
* [元学习](#元学习)
* [训练过程](#训练过程)
* [参考文献](#参考文献)

# 小样本学习问题

Few-Shot Learning (FSL)

众所周知，现在的主流的传统深度学习技术需要大量的数据来训练一个好的模型。例如典型的 MNIST 分类问题，一共有 10 个类（“0”~“9”，一共10类数字），训练集一共有 6000 个样本，平均下来每个类大约 600 个样本。

![](..\assets\img\postsimg\20200713\0.jpg)

但是我们想一下我们人类自己，我们区分 0 到 9 的数字图片的时候需要看 6000 张图片才知道怎么区分吗？很显然，不需要！这表明当前的深度学习技术和我们人类智能差距还是很大的，要想弥补这一差距，少样本学习是一个很关键的问题。

另外还有一个重要原因是如果想要构建新的数据集，还是举分类数据集为例，我们需要标记大量的数据，但是有的时候标记数据集需要某些领域的专家（例如医学图像的标记），这费时又费力，因此如果我们可以解决FSL问题，只需要每个类标记几张图片就可以高准确率的给剩余大量图片自动标记。这两方面的原因都让FSL问题很吸引人。

总结一下，传统的深度学习问题存在两个弊端：

- 与当前人脑智能存在差距，即人脑无需大量样本即可很好的完成分类等任务，而深度神经网络不行；

- 某些情况下产生大量标记样本代价很大，只能产生很小量的标记样本用以学习。

下面我们来看一张图，这张图来自论文《Optimization as a Model for Few-Shot Learning》，左边是训练集一共 5 张图片来自 5 个类，每个类只有1张图片。右边是测试集，理论上可以有任意多个图片用于测试，图中只给出了两张实例。

![](..\assets\img\postsimg\20200713\1.jpg)

如果采用传统的深度学习方法，必须能够提供大量 $D_{train}$ 样本图像，比如上图中需要提供大量鸟类、坦克、狗、人、钢琴的图像，才能训练出一个比较好的深度神经网络，使得网络能够以较高的准确率分辨 $D_{test}$ 中的图像。但是如果无法提供大量图像，那么就会出现严重的过拟合（over-fitting）问题，即因为训练的样本太少了，训练出的模型可能在训练集上效果还行，但是在测试集上面会遭遇灾难性的崩塌。或者换句话说，只有给模型提供训练集中的图片时才能正确分类，提供测试集中存在一定差异的相似图片就无法正确分类。因此，FSL问题的关键是解决过拟合 (overfitting) 的问题，一般可以采用元学习的方法来解决FSL问题。

在 FSL 中有一个术语叫做 **N-way K-shot** 问题，简单的说就是我们需要分类的样本属于 N 个类中一种，但是我们每个类训练集中的样本只有 K 个，即一共只有 N$\cdot$K 个样本的类别是已知的。上图就是一个 5-way 1-shot 的问题。

# 元学习方法

Meta-Learning

元学习的核心想法是先学习一个先验知识（prior），这个先验知识对解决 few-shot learning 问题特别有帮助。Meta-learning 中有 task 的概念，比如上面图片讲的 **5-way 1-shot 问题就是一个 task**，我们需要先学习很多很多这样的 task，然后再来解决这个新的 task 。最最最重要的一点，这是一个新的 task。

分类问题中，这个新的 task 中的类别是之前我们学习过的 task 中没有见过的！ 在 Meta-learning 中之前学习的 task 我们称为 **meta-training task**，我们遇到的新的 task 称为 **meta-testing task**。因为每一个 task 都有自己的训练集和测试集，因此为了不引起混淆，我们把 task 内部的训练集和测试集一般称为 **support set** 和 **query set**。

![](..\assets\img\postsimg\20200713\1.jpg)

还是以这张图为切入点，将几个概念之间点关系罗列如下：

在 FSL 中有一个术语叫做 **N-way K-shot** 问题，简单的说就是我们需要分类的样本属于 N 个类中一种，但是我们每个类训练集中的样本只有 K 个，即一共只有 N$\cdot$K 个样本是已知的。上图就是一个 5-way 1-shot 的问题，一共5个类，每个类只有1张图。

- N-way K-shot：样本包含 N 个类，每个类中的样本只有 K 个，一共有 N$\cdot$K 个样本
- task (meta-task)：任务，一个特定的 N-way K-shot 的任务就是一个 task
- meta-training：元训练
- meta-testing：元测试
- meta-training task: 元训练阶段的任务
  - support set：元训练阶段的训练集
  - query set：元训练阶段的测试集
- meta-testing：元测试阶段
  - training set：元测试阶段的训练集
  - test set：元测试阶段的测试集

**Few-shot Learning 是 Meta Learning 在监督学习领域的应用**。在 meta-training 阶段将数据集分解为不同的 meta task，去学习类别变化的情况下模型的泛化能力，在 meta-testing 阶段，面对全新的类别，不需要变动已有的模型，通过一步或者少数几步训练，就可以完成分类。

# 训练过程

**N-way K-shot 问题**的具体训练过程如下：

- 首先提供一个 few-shot 的数据集，该数据集一般包含了很多的类别，每个类别中又包含了很多个样本（图片）。
- 对训练集进行划分，随机选出若干类别作为训练集，剩余类别作为测试集；

 meta-train 阶段：

- 在训练集中随机抽取 **N** 个类，每个类 **K** 个样本，为支撑集（support set），剩余样本为问询集（query set）；
  - 在query set 中，剩余样本不一定全都要用到，如下图只用了5类中的2类，每类1个样本；
  - support set 和 query set 构成一个 meta-task；
- 每次采样一个 meta-task 进行训练，称为一个 episode；
- 如此反复可以选出若干个meta-task，构成一个batch；
  - 一次meta-train可以训练多干个batch（比如10000个）。

meta-test 阶段：

- 在测试集中随机抽取 N 个类别，每个类别 K 个样本，作为 train set，剩余样本作为 test set；
- 用 support set 来 fine-tune 模型；
- 用 test set 来测试模型。

![](..\assets\img\postsimg\20200713\2.1.jpg)

上图展示了 **5-way 1-shot** 的分类问题。N=5，K=1。

![](..\assets\img\postsimg\20200713\5.jpg)

上图展示了 **2-way 4-shot** 的分类问题。N=2，K=4。

上述训练过程中，每次训练（episode）都会采样得到不同 meta-task，所以总体来看，训练包含了不同的类别组合，这种机制使得模型学会不同 meta-task 中的共性部分，比如如何提取重要特征及比较样本相似等，忘掉 meta-task 中 task 相关部分。通过这种学习机制学到的模型，在面对新的未见过的 meta-task 时，也能较好地进行分类。

# 参考文献

<span id="ref1">[1]</span>  [CaoChengtai](https://blog.csdn.net/weixin_37589575). [Few-shot learning（少样本学习）和 Meta-learning（元学习）概述](https://blog.csdn.net/weixin_37589575/article/details/92801610).
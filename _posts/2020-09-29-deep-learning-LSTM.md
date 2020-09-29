---
title: 深度学习基础（LSTM）
date: 2020-09-29 09:43:19 +0800
categories: [Knowledge, DeepLearning]
tags: [academic]
math: true
---

本文介绍了 RNN （循环神经网络）和 LSTM （长短时记忆网络）的基本概念。

<!--more-->

---

- [1. RNN](#1-rnn)
  - [模型](#模型)
- [3. 参考文献](#3-参考文献)

# 1. RNN

在前馈神经网络中，信息传递是单向的[[1](#ref1)]。前馈神经网络可以看作一个复杂的函数，每次输入都是独立的，即网络的输出只依赖于当前的输入．但是在很多现实任务中，网络的输出不仅和当前时刻的输入相关，也和其过去一段时间的输出相关。．此外，前馈网络难以处理时序数据，比如视频、语音、文本等．时序数据的长度一般是不固定的，而前馈神经网络要求输入和输出的维数都是固定的，不能任意改变．因此，当处理这一类和时序数据相关的问题时，就需要一种能力更强的模型．

循环神经网络（Recurrent Neural Network，RNN）是一类具有短期记忆能力的神经网络．在循环神经网络中，神经元不但可以接受其他神经元的信息，也可以接受自身的信息，形成具有环路的网络结构．和前馈神经网络相比，循环神经网络更加符合生物神经网络的结构．循环神经网络已经被广泛应用在语音识别、语言模型以及自然语言生成等任务上．

## 模型

循环神经网络（Recurrent Neural Network，RNN）通过使用带自反馈的神经元，能够处理任意长度的时序数据。

![rnn](../assets/img/postsimg/20200929/1.jpg)

# 3. 参考文献

<span id="ref1">[1]</span>  邱锡鹏. 神经网络与深度学习.

<span id="ref2">[2]</span> Rudolf Kruse. [Fuzzy neural network](http://www.scholarpedia.org/article/Fuzzy_neural_network).

<span id="ref3">[3]</span> Milan Mares. [Fuzzy Sets](http://www.scholarpedia.org/article/Fuzzy_systems).

[4] L.A. Zadeh. [Fuzzy sets](https://www.sciencedirect.com/science/article/pii/S001999586590241X).
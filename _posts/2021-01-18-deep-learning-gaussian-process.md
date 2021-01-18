---
title: 深度学习基础（高斯过程）
date: 2021-01-18 16:21:49 +0800
categories: [Academic, Knowledge]
tags: [deeplearning]
math: true
---

本文介绍了高斯过程，包括高斯函数、多元高斯分布、高斯过程。

<!--more-->

 ---
 
- [1. 简介](#1-简介)
- [2. 下载与安装](#2-下载与安装)
- [3. 配置 R 开发环境](#3-配置-r-开发环境)
  - [3.1. 安装 R 语言](#31-安装-r-语言)
  - [3.2. 安装 LanguageServer](#32-安装-languageserver)
  - [3.3. 安装扩展](#33-安装扩展)
  - [测试 R 环境](#测试-r-环境)
- [4. 参考文献](#4-参考文献)

# 1. 高斯函数

标准高斯函数为

$$
f(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}
$$

函数图像为

这个函数描述了变量 $x$ 的一种分布特性，变量x的分布有如下特点：

- 均值 = 0
- 方差 = 1
- 概率密度和 = 1

一元高斯函数的一版形式为

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

若令

$$
z = \frac{x-\mu}{\sigma}
$$

称这个过程为标准化，那么 $z\sim N(0,1)$。

# 2. 多元高斯分布


# 3. 高斯过程


# 4. 参考文献

[1] bingjianing. [多元高斯分布（The Multivariate normal distribution）](https://www.cnblogs.com/bingjianing/p/9117330.html)

[2] 论智. [图文详解高斯过程（一）——含代码](https://zhuanlan.zhihu.com/p/32152162)
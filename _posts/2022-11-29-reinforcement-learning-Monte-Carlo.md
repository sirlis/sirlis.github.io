---
title: 强化学习（model-free 蒙特卡洛法）
date: 2022-11-29 17:07:19 +0800
categories: [Academic, Knowledge]
tags: [python,reinforcementlearning]
math: true
---

本文介绍了强化学习的 model-free 方法——蒙特卡洛法。

<!--more-->

---

## 1. 引言

在前面的 model-based 动态规划方法中，我们假设已知模型的动态特性 $p(s^\prime,r \vert s,a)$，此时可以对下一步的状态和汇报做出预测。而在很多实际案例中，我们无法得知模型的动态特性，此时动态规划方法就不适用了。


## 2. 参考文献

[1] shuhuai008. [bilibili蒙特卡洛方法-强化学习-合集](https://space.bilibili.com/97068901/channel/collectiondetail?sid=196314)

[2] 韵尘. [知乎：5.1 —— 蒙特卡洛预测（Monte Carlo Prediction）](https://zhuanlan.zhihu.com/p/538564739)
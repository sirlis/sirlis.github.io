---
title: 强化学习（时序差分法）
date: 2022-12-18 14:59:19 +0800
categories: [Academic, Knowledge]
tags: [python,reinforcement learning]
math: true
---

本文介绍了强化学习的时序差分法（Temporal-Difference, TD）。

<!--more-->

---

## 引言

在前面的的介绍中，我们分别介绍了两种基于价值的方法，动态规划法和蒙特卡洛法。本节介绍第三种基于价值的方法：时序差分法。

首先回顾一下价值函数的等式：

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi[G_t\vert S_t=s] & {MC}\\
&= \mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t=s] & {TD}\\
&= \sum_a\pi(a\vert s) \sum_{s^\prime,r}p(r^\prime,r \vert s,a)(r+\gamma v_\pi(s^\prime)) & {DP}\
\end{aligned}
$$

其中：
- DP：自举。更新 $v_{k+1}$ 时采用上一步的 $v_k$ 进行组装，**缺点：环境动态特性必须已知**；
- MC：采样。依据大数定律，让样本均值逼近期望，**缺点：必须完整采集一幕**；

### 测试1

#### 测试11

#### 测试12

#### 测试13

## 测试2

### 测试22

### 测试323

## 参考文献

[1] 刘建平Pinard. [强化学习（三）用动态规划（DP）求解](https://www.cnblogs.com/pinard/p/9463815.html).

[2] Zeal. [知乎：强化学习二：策略迭代法](https://zhuanlan.zhihu.com/p/358464793)

[3] shuhuai008. [bilibili【强化学习】动态规划【白板推导系列】](https://www.bilibili.com/video/BV1nV411k7ve)

[4] 韵尘. [知乎：4.2 —— 策略改进（Policy Improvement）](https://zhuanlan.zhihu.com/p/537229275)（含收敛性证明）
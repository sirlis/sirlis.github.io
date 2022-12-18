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

## 1. 引言

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
- TD：

因此时序差分法（TD0），是在 MC 的基础上，不走完整个序列，而是只走**一步**

$$
\begin{aligned}
MC:\quad & v(S_t) \leftarrow v(S_t)+\alpha(G_t-V(S_t)) \\
TD0:\quad & v(S_t) \leftarrow v(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1})-V(S_t))
\end{aligned}
$$


## 参考文献

[3] shuhuai008. [bilibili【【强化学习】 时序差分-策略评估](https://www.bilibili.com/video/BV1wS4y1F7zn)

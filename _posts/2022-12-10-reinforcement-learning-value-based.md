---
title: 强化学习（model-free value-based）
date: 2022-12-10 11:24:19 +0800
categories: [Academic, Knowledge]
tags: [python,reinforcement learning]
math: true
---

本文介绍了强化学习的 model-free 方法。 model-free 方法主要包括 value-based 方法和 policy-based 方法。本文重点介绍 value-based 方法，包括 SARSA 和 Q-Learning。

<!--more-->

---

## 1. 引言

在前面的 model-based 动态规划方法中，我们假设已知模型的动态特性 $p(s^\prime,r \vert s,a)$，此时可以对下一步的状态和汇报做出预测。而在很多实际案例中，我们无法得知模型的动态特性，此时动态规划方法就不适用了。


## 2. 参考文献

[1] 刘建平Pinard. [强化学习（三）用动态规划（DP）求解](https://www.cnblogs.com/pinard/p/9463815.html).

[2] Zeal. [知乎：强化学习二：策略迭代法](https://zhuanlan.zhihu.com/p/358464793)

[3] shuhuai008. [bilibili【强化学习】动态规划【白板推导系列】](https://www.bilibili.com/video/BV1nV411k7ve)

[4] 韵尘. [知乎：4.2 —— 策略改进（Policy Improvement）](https://zhuanlan.zhihu.com/p/537229275)（含收敛性证明）
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
MC:\quad & v(S_t) \leftarrow v(S_t)+\alpha(G_t-v(S_t)) \\
TD0:\quad & v(S_t) \leftarrow v(S_t)+\alpha(R_{t+1}+\gamma v(S_{t+1})-v(S_t))
\end{aligned}
$$

### 同轨策略下的时序差分控制（SARSA）

$$
\begin{aligned}
V(S_t) & \leftarrow V(S_t)+\alpha(\underbrace{R_{t+1}}_{采样}+\gamma \underbrace{V(S_{t+1})}_{自举}-V(S_t))\\
Q(S_t,A_t) & \leftarrow Q(S_t,A_t)+\alpha(\underbrace{R_{t+1}}_{采样}+\gamma \underbrace{Q(S_{t+1},A_{t+1})}_{脑采}-Q(S_t,A_t))\\
\end{aligned}
$$

上述更新方式即为 **SARSA。**

伪代码如下：


Initialize Q(s,a) arbitrarily
Repeat (for each episode):
    $\qquad$Initialize s
    $\qquad$Repeat (for each step of episode):
        $\qquad$$\qquad$Choose a from s using policy derived from Q (e.g., $\epsilon$-greedy)
        $\qquad$$\qquad$Take action a, observe r, s'
        $\qquad$$\qquad$<font color=red>Choose $a^\prime$ from $s^\prime$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)</font>
        $\qquad$$\qquad$$\color{red}{Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma Q(s^\prime,a^\prime)-Q(s,a)]}$
        $\qquad$$\qquad$$\color{red}{s\leftarrow}s^\prime; a\leftarrow a^\prime$
    $\qquad$until $s$ is terminal

上述更新方式又被称为同轨策略（on-policy），因为其采样和更新的均为同一个策略。

### 离轨策略下的时序差分控制

实际上，在 $s^\prime$ 状态下，存在一个确定性策略

$$
a^*=\pi(s^\prime) = argmax_a\; Q(s^\prime, a)
$$

此时有

$$
max_{a^\prime}\; Q(s^\prime,a^\prime) = Q(s^\prime, a^*)
$$

更新 $Q(s,a)$ 时，不再根据当前策略进行采样，而是使用这个确定性策略。这种更新方式即为 **Q-Learning**。

伪代码如下：


Initialize Q(s,a) arbitrarily
Repeat (for each episode):
    $\qquad$Initialize s
    $\qquad$Repeat (for each step of episode):
        $\qquad$$\qquad$Choose a from s using policy derived from $Q$ (e.g., $\epsilon$-greedy)
        $\qquad$$\qquad$Take action a, observe r, s'
        $\qquad$$\qquad$$\color{red}{Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma 
        \;Q(s^\prime,a^*)-Q(s,a)]}$
        $\qquad$$\qquad$$\color{red}{s\leftarrow}s^\prime$
    $\qquad$until $s$ is terminal

此时，更新时使用（采样得到动作）的策略 $\pi$ 并不是我们待更新的策略，因此被称为离轨策略（off-policy）。

### 期望SARSA

我们可以对 SARSA 进行改进，不再进行采样得到动作 $a^\prime$，而是对 $Q$ 进行加权平均，此时

$$
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma \mathbb{E}_\pi[Q(s^\prime,a^\prime)\vert s^\prime]-Q(s,a)]
$$

## 参考文献

[1] shuhuai008. [bilibili【强化学习】(SARSA) 时序差分-同轨策略TD控制](https://www.bilibili.com/video/BV1BS4y1r7cm)

[2] 莫烦. [什么是 Sarsa (强化学习)](https://zhuanlan.zhihu.com/p/24860793)

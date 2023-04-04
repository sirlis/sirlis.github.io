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


回顾强化学习的目标：价值估计（预测问题）和策略寻优（控制问题）。在前面的的介绍中，我们分别介绍了两种基于价值的方法，动态规划法和蒙特卡洛法：

- **动态规划法**（DP）：是 model-based 方法，包含策略评估和策略改进两步，策略评估用来进行价值估计（即预测问题），策略改进用来进行策略寻优（控制问题）。
- **蒙特卡洛法**（MC）：是 model-free 方法，因为一般情况下我们无法得到具体模型（状态转移概率），因此通过采样完整序列后，通过 $G_t$ 来进行策略评估（价值估计）。

本节介绍第三种基于价值的方法：时序差分法（TD）。首先回顾一下价值函数的等式：

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi[G_t\vert S_t=s] & {MC}\\
&= \mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t=s] & {TD}\\
&= \sum_a\pi(a\vert s) \sum_{s^\prime,r}p(r^\prime,r \vert s,a)(r+\gamma v_\pi(s^\prime)) & {DP}\
\end{aligned}
$$

可以看出，基于价值的方法可以根据价值函数的等式不同来划分，其中：
- 动态规划（DP）：是一种自举的方法。更新 $v_{k+1}$ 时采用上一步的 $v_k$ 进行组装，**缺点：环境动态特性必须已知**；
- 蒙特卡洛（MC）：是一种采样的方法。依据大数定律，让样本均值逼近期望，**缺点：必须完整采集一幕**；
- 时序差分（TD）：本章节介绍的方法。

蒙特卡洛方法必须要等整个序列结束之后才能计算得到这一次的回报 $G_{t}$，而时序差分（Temporal Difference, TD）只需要当前步结束即可进行计算。具体而言，是将 $G_t$ 根据其定义进行一步展开

$$
\begin{aligned}
G_t \doteq \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} &= R_{t+1}+\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+2}\\
&= R_{t+1}+\gamma V_{t+1}    
\end{aligned}
$$

然后带入蒙特卡洛的价值函数增量更新策略，即

$$
\begin{aligned}
MC:\quad & v(S_t) \leftarrow v(S_t)+\alpha({\color{red}G_t}-v(S_t)) \\
TD:\quad & v(S_t) \leftarrow v(S_t)+\alpha({\color{red}R_{t+1}+\gamma v(S_{t+1})}-v(S_t))
\end{aligned}
$$

也就是说，时序差分算法用当前获得的奖励加上下一个状态的价值估计来作为在当前状态会获得的回报。这里我们将动态路径规划节的$\frac{1}{N(s)}$替换成了 $\alpha$ ，表示对价值估计更新的步长。可以将 $\alpha$ 取为一个常数，此时更新方式不再像蒙特卡洛方法那样严格地取期望。

时序差分算法用到了 $V(s_{t+1})$ 的估计值，可以证明它最终收敛到策略 $\pi$ 的价值函数，我们在此不对此进行展开说明。


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
        $\qquad$$\qquad$$\color{red}{Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma \mathop{\text{max}}\limits_{a}
        Q(s^\prime,a)-Q(s,a)]}$
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

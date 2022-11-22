---
title: 强化学习（马尔可夫决策过程）
date: 2022-11-09 12:36:19 +0800
categories: [Academic, Knowledge]
tags: [python]
math: true
---

本文介绍了强化学习的基本概念和模型，主要包括马尔可夫过程、马尔可夫奖励过程和马尔可夫决策过程。

<!--more-->

---

- [1. 强化学习](#1-强化学习)
  - [1.1. 状态和观测](#11-状态和观测)
  - [1.2. 动作空间](#12-动作空间)
  - [1.3. 策略](#13-策略)
- [2. 马尔可夫过程（MP）](#2-马尔可夫过程mp)
- [3. 马尔可夫奖励过程（MRP）](#3-马尔可夫奖励过程mrp)
  - [3.1. 奖励（Reward）](#31-奖励reward)
  - [3.2. 回报（Return）](#32-回报return)
  - [3.3. 价值函数（Value Function）](#33-价值函数value-function)
  - [3.4. 贝尔曼方程（Bellman Equation）](#34-贝尔曼方程bellman-equation)
- [4. 马尔可夫决策过程（MDP）](#4-马尔可夫决策过程mdp)
  - [4.1. 动作（Action）](#41-动作action)
  - [4.2. 策略（Policy）](#42-策略policy)
  - [4.3. 动态特性](#43-动态特性)
  - [4.4. 价值函数（Value Function）](#44-价值函数value-function)
    - [4.4.1. 状态价值函数](#441-状态价值函数)
    - [4.4.2. 动作价值函数](#442-动作价值函数)
    - [4.4.3. 回溯图与回溯操作](#443-回溯图与回溯操作)
- [5. 参考文献](#5-参考文献)

## 1. 强化学习


强化学习是机器学习领域之一，受到行为心理学的启发，主要关注智能体如何在环境中采取不同的行动，以最大限度地提高累积奖励。强化学习是除了监督学习和非监督学习之外的第三种基本的机器学习方法。与监督学习不同的是，强化学习不需要带标签的输入输出对，同时也无需对非最优解的精确地纠正。

强化学习主要由智能体（Agent）、环境（Environment）、状态（State）、动作（Action）、奖励（Reward）组成。智能体执行了某个动作后，环境将会转换到一个新的状态，对于该新的状态环境会给出奖励信号（正奖励或者负奖励）。随后，智能体根据新的状态和环境反馈的奖励，按照一定的策略执行新的动作。上述过程为智能体和环境通过状态、动作、奖励进行交互的方式。

在每个时刻 $t$，智能体观察到所在的环境状态记为 $s_t$，，并再次基础上选择一个动作 $a_t$。作为动作的结果，智能体接收到一个数值化的奖励 $r_{t+1}$，并转移到一个新的状态 $s_{t+1}$ 。

![强化学习示意图](/assets/img/postsimg/20221109/0-reinforcement-learning-basic-diagram.jpg)


如此反复迭代交互，可以得到一个序列或轨迹（trajectory）。

强化学习的目标：**最大化智能体接收到的标量信号（奖励）累积和的概率期望值** $G$

$$
G = r_1+r_2+...+r_n
$$

### 1.1. 状态和观测

「状态」$s$ 是对环境当前所处环境的完整描述，对于状态来说环境的所有信息都是可知的。而「观测」$o$ 则是一个状态的部分描述，可能会忽略一些信息。

在深度强化学习中，我们通常会使用实值向量`vector`、矩阵 `matrix` 或者高维张量 `tensor` 来表示状态和观测。比如，视觉的观测可以表示为像素值构成的 RGB 矩阵，机器人的状态则可以表示为其关节角度和速度。

当智能体能够观测到环境的全部状态时，这样的环境是可完全观测的 `fully observed`；当智能体只能观测到部分状态时，这样的环境称为可部分观测 `partially observed`。

>「注」：在强化学习的公式中我们经常会看到表示状态的符号 $s$，但是实际上更准确的用法应该是使用表示观测的符号 $o$。比如，当我们探讨智能体如果进行动作决策时，公式中通常会说动作是基于状态的，但是「实际上动作是基于观测的」，因为智能体是无法直接感知到状态的。为了遵循惯例，之后的公式仍然会使用符号 $s$。
❞

### 1.2. 动作空间

「**给定的环境中有效的动作集合称为动作空间**」。有些环境中（比如 Atari 和 Go），「动作空间是离散的」`discrete`，也就是说智能体的动作数量是有限的；而有些环境中（比如机器人控制），「动作空间是连续的」`continuous`，这些空间中动作通常用实值向量表示。

「**动作空间是离散的还是连续的**」在强化学习问题中非常重要，有些方法只适合用于其中一个场景，所以这点需要特别关注。

### 1.3. 策略

## 2. 马尔可夫过程（MP）

马尔可夫过程（Markov Process）：

- 在一个时序过程中，如果 $t＋1$ 时刻的状态仅取决于 $t$ 时刻的状态 $s_t$ 而与 $t$ 时刻之前的任何状态都无关时，则认为 $t$ 时刻的状态 $s_t$ 具有马尔可夫性(Markov property)；

- 若过程中的每一个状态都具有马尔科夫性，则这个过程具备马尔科夫性。具备了马尔科夫性的随机过程称为马尔科夫过程，又称马尔科夫链（Markov chain）。

表述一个马尔可夫过程，只需要二元组 $<S,P>$ 即可。其中，$S$ 为状态，$P$ 为不同状态间的转移概率，如状态 $s$ 到 $s^\prime$ 的状态转移概率

$$
P_{s s^\prime}=\mathbb{P}[S_{t+1}=s^\prime \vert s_{t}=s]
$$

从上式也可以看到，下一时刻的状态$S_{t+1}$ 仅与当前时刻的状态 $S_t$ 有关，而与 $S_1,\cdots,S_{t-1}$ 无关。注意：这里的记号非常严谨， $S_{t}, S_{t+1}$ 代表某一时刻的状态，而 $s,s^\prime$ 代表某一种具体的状态类型。

为了描述整个状态空间中不同类型状态之间的关系，用矩阵表示，即为：

$$
P=
\begin{bmatrix}
P(s_1\vert s_1) & P(s_2\vert s_1) &\dots & P(s_n\vert s_1)\\
P(s_1\vert s_2) & P(s_2\vert s_2) &\dots & P(s_n\vert s_2)\\
\vdots & \vdots & \ddots &\vdots\\
P(s_1\vert s_n) & P(s_2\vert s_n)&\dots & P(s_n\vert s_n)\\
\end{bmatrix}
$$

其中，$P(s_j\vert s_i)$ 表示状态从 $s_i$ 转移到 $s_j$ 的状态转移概率，显然状态转移概率矩阵 $P$ 的规模是所有状态类型数 $n$ 的平方。

以下图为例：

![马尔可夫过程](/assets/img/postsimg/20221109/1-mp.png)

不难写出状态转移概率矩阵：

$$
\begin{aligned}
&\quad C1\;\;\;C2\;\;\;C3\;\;\;Pass\;Pub\;FB\;\;Sleep\\
P=
\begin{array}{r}
    C1\\
    C2\\
    C3\\
    Pass\\
    Pub \\
    FB\\
    Sleep
\end{array}
&\begin{bmatrix}
    0& 0.5 &0&0&0&0.5&0\\
    0&0&0.8&0&0&0&0.2\\
    0&0&0&0.6&0.4&0&0\\
    0&0&0&0&0&0&1\\
    0.2&0.4&0.4&0&0&0&0\\
    0.1&0&0&0&0&0.9&0\\
    0&0&0&0&0&0&1\\
\end{bmatrix}
\end{aligned}
$$

马尔可夫过程中的三个概念：

- **状态序列**（episode）：一个状态转换过程，包含很多状态。如：$s_1$-$s_2$-$s_3-...$；
- **完整的状态序列**（complete episode）：状态序列的最后一个状态是终止状态；
- **采样**（sample）：从符合马尔科夫过程给定的状态转移概率矩阵生成一个状态序列的过程。

## 3. 马尔可夫奖励过程（MRP）

马尔可夫奖励过程（Markov Reward Process，MRP）就是在马尔可夫过程中增加了奖励（收益） $R$ 项和奖励因子 $\gamma$。

表述一个马尔可夫奖励过程需要**四元组** $<S,P,R,\gamma>$，其中：

- $S$ 是一个有限状态集；
- $P$ 是集合中状态转移概率矩阵：$P_{s s^\prime}=\mathbb{P}[S_{t+1}=s^\prime \vert S_t = s]$；
- $R$ 是奖励（收益）函数：$R_t = \mathbb{E}[R_{t+1}\vert S_t=s]$
- $\gamma$ 是折扣因子：$\gamma \in [0,1]$。

如下图所示

![马尔可夫奖励过程](/assets/img/postsimg/20221109/2-mrp.png)

### 3.1. 奖励（Reward）

某时刻 $t$ 的奖励 $R_t$ 定义为**从该状态转移到所有可能状态的奖励的期望**。

通常约定，某一时刻 $t$ 处在状态 $s_t$ 下奖励，是在下一个时刻 $t+1$ 获得的，即离开当前状态时获得奖励

$$
R_t = \mathbb{E}[R_{t+1}\vert S_t=s_t]
$$

> 为什么是 $R_{t+1}$ 而不是 $R_t$？因为约定离开这个状态才能获得奖励而不是进入这个状态即获得奖励。因为奖励是智能体采取动作之后离开 $S_t$ 时获得的，同时进入下一时刻状态 $S_{t+1}$。在后面的马尔可夫决策过程中，使用 $R_{t+1}$ 来表示 $A_t$ 导致的奖励，即强调下一时刻的收益和下一时刻的状态是被环境一起决定的。

举例说明：当学生处在第一节课（Class 1）时，之后参加第2节课（Class 2）获得的 Reward 是 $-2$，若之后上网浏览 Facebook 获得的 Reward 也是 $-2$。

此时，强化学习的目标：**使得智能体收到的累计奖励最大化。**

奖励是即时的，是 $t$ 时刻状态转移到 $t+1$ 时刻状态时的瞬时反馈信息。但在实际过程中，$t$ 时刻状态 $s_t$ 可能对后续所有状态产生深远影响，而不是单独对 $t+1$ 时刻状态产生影响。因此需要引入后面的新概念：回报。

### 3.2. 回报（Return）

某时刻 $t$ 的回报（收益）定义为**从时刻 $t$ 开始的 ==采样一条== 状态序列得到的所有奖励的折扣和**：

$$
G_t\doteq R_{t+1}+\gamma R_{t+2}+...=\sum_{k=0}^\infty \gamma^k R_{t+k+1}
$$

越往后得到的奖励，折扣得越多。设置折扣因子的原因如下：

- 数学表达的方便，这也是最重要的
- 避免陷入无限循环
- 远期利益具有一定的不确定性
- 在金融学上，立即的回报相对于延迟的汇报能够获得更多的利益
- 符合人类更看重眼前利益的性格

折扣因子可以作为强化学习的一个超参数来进行调整，折扣因子不同就会得到不同行为的智能体。折扣因子接近 $0$ 则表明趋向于“近视”性评估；接近 $1$ 则表明偏重考虑远期的利益。

假设从 $Class 1$ 状态开始到 $Sleep$ 状态终止，折扣因子 $\gamma = 0.5$，采样两条序列计算回报如下：

```
C1 - C2 - C3 - Pass - Sleep
G_class1 = -2 + 1/2 × (-2) + 1/4 × (-2) + 1/8 × (+10) = -2.25
C1 - FB - FB - C1 - C2 - Sleep
G_class1 = -2 + 1/2 × (-1) + 1/4 × (-1) + 1/8 × (-2) + 1/16 ×(-2) = -3.125
```

此时，强化学习的目标：**使得智能体收到的回报最大化。**

相邻时刻的回报可用如下递归形式联系起来，这对于强化学习的理论和算法至关重要

$$
G_t=R_{t+1}+\gamma R_{t+2}+...
$$

回报值是针对一次完整的采样序列的结果，存在很大的样本偏差。即 $G(s)$ 是从 $t$ 时刻的状态到终止状态的一条状态转移序列的回报值，但从 $t$ 时刻的状态到终止状态的马尔可夫链不止一条，每一条都有对应的采样概率和回报。对于复杂问题而言，要想精确的计算出 $G(s)$ 是几乎不可能的，因为无法穷举所有序列。

为了能够评估状态的好坏，引入新概念：价值函数。

### 3.3. 价值函数（Value Function）

**价值函数**（Value）：**从某个状态 $s_t$ 开始的回报的期望，也即从某个状态 $s_t$ 开始采样无数条完整状态序列的回报的平均值**，即

$$
V(s_t) = \mathbb{E}[G_t \vert S_t=s_t]
$$

对于马尔可夫奖励过程，价值函数即为状态价值函数。

以前面的例子，如果仅观测到两个序列，那么在状态 Class 1 处的学生的值函数就是 2 个回报值除以 2 即可。

```
v(Class1) = ( (-2.25) + (-3.125))  ÷ 2 = -2.6875
```

状态值函数的引入解决了回报 $G(s)$ 路径有很多条，不容易优化的问题，**将其转化为期望，变成固定标量了**。

但状态值函数也不好算，因为在计算某个状态时候需要使用到将来所有状态的 $G(s)$。为了便于计算，对价值函数进行展开

$$
\begin{aligned}
V(s_t) &= \mathbb{E}[G_t \vert S_t=s_t]\\
&=\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...\vert S_t=s_t]\\
&=\mathbb{E}[R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}+...)\vert S_t=s_t]\\
&=\mathbb{E}[R_{t+1}+\gamma G_{t+1}\vert S_t=s_t]\\
&=\mathbb{E}[R_{t+1}\vert S_t=s]+\gamma \mathbb{E}[G_{t+1}\vert S_t=s_t]\\
&=R_{s}+\gamma \mathbb{E}[G_{t+1}\vert S_t=s_t]
\end{aligned}
$$

上式中，第一项 $R_s$ 对应即时奖励，第二项则代表了长期的潜在奖励。可以看出，长期潜在奖励的计算需与下一时刻状态对应回报的期望。
然而，未来时刻的状态及其回报是不确定的，即

$$
\mathbb{E}[G_{t+1}\vert S_t=s_t]
$$

很难求解。因此直接计算价值函数是不现实的。下面介绍贝尔曼方程来计算价值函数。

### 3.4. 贝尔曼方程（Bellman Equation）

**[ 推导 1 ]：**

> - **定义**：如果 $X$ 和 $Y$ 都是离散型随机变量，则条件期望（Conditional Expectation）定义为
>   $\mathbb{E}[Y\vert X=x]=\sum_y yP(Y=y\vert X=x)$
> - **定义**：如果 $X$ 是随机变量，其期望为 $\mathbb{E}[X]$，$Y$ 为相同概率空间上的任意随机变量，则有全期望（Total Expectation）公式
>   $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X\vert Y]]$

现证明（主要证明第一个等式）

$$
\mathbb{E}[G_{t+1}\vert S_t=s_t] = \mathbb{E}[\mathbb{E}[G_{t+1}\vert S_{t+1}]\vert S_t=s_t] = \mathbb{E}[V(s_{t+1})\vert S_t=s_t]
$$

为了推导简便，另 $s_{t+1} = s^\prime$，$s_t=s$，有

$$
\begin{aligned}
\mathbb{E}[\mathbb{E}[G_{t+1}\vert S_{t+1}]\vert S_t=s] &= \mathbb{E}\left[\sum_{g^\prime}g^{\prime}P(G(s^\prime)=g^{\prime}\vert S_{t+1})\vert s\right]\quad (条件期望)\\
&=\sum_{s^\prime} \sum_{g^\prime}g^{\prime}P(G(s^\prime)=g^{\prime}\vert S_{t+1}=s^\prime, s)P(S_{t+1}=s^\prime\vert s)\\
&=\sum_{s^\prime} \sum_{g^\prime}g^{\prime} \frac{P(G(s^\prime)=g^{\prime}\vert S_{t+1}=s^\prime, s)P(S_{t+1}=s^\prime\vert s)\cdot P(s)}{P(s)} \\
&=\sum_{s^\prime} \sum_{g^\prime}g^{\prime} \frac{P(G(s^\prime)=g^{\prime}\vert S_{t+1}=s^\prime, s)P(S_{t+1}=s^\prime, s)}{P(s)} \\
&=\sum_{s^\prime} \sum_{g^\prime}g^{\prime} \frac{P(G(s^\prime)=g^{\prime}, S_{t+1}=s^\prime, s)}{P(s)} \\
&=\sum_{s^\prime} \sum_{g^\prime}g^{\prime} P(G(s^\prime)=g^{\prime}, S_{t+1}=s^\prime \vert s) \\
&=\sum_{g^\prime} \sum_{s^\prime}g^{\prime} P(G(s^\prime)=g^{\prime}, S_{t+1}=s^\prime \vert s) \\
&=\sum_{g^\prime}g^{\prime} P(G(s^\prime)=g^{\prime} \vert s) \\
&=\mathbb{E}[G(s^\prime)\vert s]=\mathbb{E}[G_{t+1}\vert s_t]
\end{aligned}
$$

得证。则当前时刻的状态价值函数

$$
\begin{aligned}
V(s_t)&=R_{s}+\gamma \mathbb{E}[G_{t+1}\vert S_t=s_t]\\
&=R_{s}+\gamma \mathbb{E}[V(s_{t+1})\vert S_t=s_t]\\
&=R_{s}+\gamma \sum_{s_{t+1}\in S} V(s_{t+1})P(s_{t+1}\vert s_t)
\end{aligned}
$$

上式即为马尔可夫奖励过程的贝尔曼方程。

**[ 推导 2 ]：** （可能不对？）

对后项进行全概率展开

$$
\begin{aligned}
\gamma \mathbb{E}[G_{t+1}\vert S_t=s_t] &= \gamma \sum_{s_{t+1}\in S}\mathbb{E}[G_{t+1}\vert S_{t+1}=s_{t+1}]P(s_{t+1}\vert s_t)\\
&= \gamma \sum_{s_{t+1}\in S} V(s_{t+1})P(s_{t+1}\vert s_t)
\end{aligned}
$$

上面第二步是因为（根据价值的定义）

$$
V(s_{t+1}) = \mathbb{E}[G_{t+1}  \vert S_{t+1} = s_{t+1}]
$$

最终得到

$$
V(s_t) = R_{s}+\gamma \sum_{s_{t+1}\in S} V(s_{t+1})P(s_{t+1}\vert s_t)
$$

即为马尔可夫奖励过程的贝尔曼方程。

---

**贝尔曼方程刻画了当前状态 $s_t$ 和下一个状态 $s_{t+1}$ 之间的关系**。可以看出，当前状态的价值函数可以通过下一个状态的价值函数来迭代计算。

若将马尔可夫奖励过程的状态构成 $n$ 维状态空间，贝尔曼方程可以写成矩阵形式

$$
\begin{aligned}
\boldsymbol{V} &= \boldsymbol{R}+\gamma \boldsymbol{P} \boldsymbol{V}\\
\begin{bmatrix}
    V(s_1)\\
    V(s_2)\\
    \vdots\\
    V(s_n)
\end{bmatrix} &=
\begin{bmatrix}
    R(s_1)\\
    R(s_2)\\
    \vdots\\
    R(s_n)
\end{bmatrix}
+\gamma
\begin{bmatrix}
P(s_1\vert s_1) & P(s_2\vert s_1)& \cdots & P(s_n\vert s_1)\\    
P(s_1\vert s_2) & P(s_2\vert s_2)& \cdots & P(s_n\vert s_2)\\    
\vdots & \vdots & \ddots & \vdots\\    
P(s_1\vert s_n) & P(s_2\vert s_n)& \cdots & P(s_n\vert s_n)\\    
\end{bmatrix}
\begin{bmatrix}
    V(s_1)\\
    V(s_2)\\
    \vdots\\
    V(s_n)
\end{bmatrix}
\end{aligned}
$$

上述是个线性方程组，可直接得到解析解

$$
\boldsymbol{V} = (\boldsymbol{I}-\gamma\boldsymbol{P})^{-1}\boldsymbol{R}
$$

需要注意的是，矩阵求逆的复杂度为 $O(n^3)$，因此直接求解仅适用于状态空间规模小的问题。状态空间规模大的问题的求解通常使用迭代法。常用的迭代方法有：动态规划(Dynamic Programming)、蒙特卡洛评估(Monte-Carlo evaluation)、时序差分学(Temporal-Difference)等。

## 4. 马尔可夫决策过程（MDP）

马尔可夫决策过程是在马尔可夫奖励过程的基础上加入了决策，即增加了动作。其定义为：

马尔科夫决策过程是一个**五元组** $<S,A,P,R,\gamma>$，其中

- $S$ 是有限数量的状态集
- $A$ 是有限数量的动作集
- $P$ 是状态转移概率，$P_{ss^\prime}^a=\mathbb{P}[S_{t+1} = s^\prime \vert S_t = s, A_t=a]$
- $R$ 是一个奖励函数，$R_{s}^a=\mathbb{E}[R_{t+1} \vert S_t = s, A_t=a]$
- $\gamma$ 是一个折扣因子，$\gamma \in [0, 1]$

从上面定义可以看出，马尔可夫决策过程的状态转移概率和奖励函数不仅取决于智能体当前状体，**还取决于智能体选取的动作**。而马尔可夫奖励过程仅取决于当前状态。

### 4.1. 动作（Action）

以下图为例：

![马尔可夫决策过程举例](/assets/img/postsimg/20221109/3-mdp.png)

图中红色的文字表示学生采取的动作，而不是 MRP 时的状态名。对比之前的学生 MRP 示例可以发现，即时奖励与动作有关了，同一个状态下采取不同的动作得到的即时奖励是不一样的。

由于引入了动作，容易与状态名称混淆，因此此图没有给出各状态的名称；此图还把 Pass 和 Sleep 状态合并成一个终止状态；另外当选择”去查阅文献（Pub）”这个动作时，主动进入了一个临时状态（图中用黑色小实点表示），随后被动的被环境按照其动力学分配到另外三个状态，也就是说此时智能体没有选择权决定去哪一个状态。

可以看出，状态转移概率 $P_{ss^\prime}^a$ 和奖励函数 $R_{s}^a$ 均与当前状态 $s$ 下采取的动作 $a$ 有关。由于动作的选取不是固定的，因此引入新概念：策略。

### 4.2. 策略（Policy）

**策略 $\pi(a\vert s)$ 是从状态 $s$ 到每个动作 $a$ 的选择概率之间的映射** ，即

$$
\pi(a\vert s) = \mathbb{P}[A_t=a\vert S_t=s]
$$

一个策略完整地定义了智能体的行为方式，即策略定义了智能体在各种状态下可能采取的动作，以及在各种状态下采取各种动作的概率。MDP的策略仅与当前的状态有关，与历史信息无关；同时某一确定的策略是静态的，与时间无关；但是个体可以随着时间更新策略。

给定一个马尔可夫决策过程 $<S,A,P,R,\gamma>$ 和一个策略 $\pi$ 后，相应的状态转移概率 $P_{ss^\prime}^\pi$ 和奖励函数 $R_{s}^\pi$ 可更新描述如下

$$
\begin{aligned}
    P_{ss^\prime}^\pi &= \sum_{a\in A}\pi(a\vert s)P_{ss^\prime}^a\\
    R_{s}^\pi &= \sum_{a\in A}\pi(a\vert s)R_{s}^a
\end{aligned}
$$

对应的 $<S,P^\pi,R^\pi,\gamma>$ 是一个马尔可夫奖励过程， $<S,P^\pi>$ 是一个马尔可夫过程。

### 4.3. 动态特性

在有限 MDP 中，状态、动作和奖励的集合（$S, A, R$）都只有有限个元素。在这种情况下，随机变量 $R_t$ 和 $S_t$ 具有定义明确的离散概率分布，并且之依赖于前一时刻的状态和动作。也就是说，给定前一时刻的状态和动作的值时，这些随机变量的特定值 $s^\prime \in S, r^\prime \in R$ 在 $t$ 时刻出现的概率为

$$
p(s^\prime,r \vert s,a) \doteq \mathbb{P}\{S_t=s^\prime, R_t=r \vert S_{t-1}=s, A_{t-1}=a\} 
$$

函数 $p$ 定义了 MDP 的**动态特性**。动态特性函数是一个描述 $t-1$ 和 $t$ 前后两个相邻时刻的随机变量间动态关系的条件概率。

在 MDP 中，由 $p$ 给出的概率完全刻画了环境的动态特性，即$S_t,R_t$ 的每个可能的值出现的概率只取决于前一个状态 $S_{t-1}$ 和动作 $A_{t-1}$，且与更早时刻的状态和动作无关（马尔可夫性）。

从四参数动态函数 $p$ 中，可以计算出关于环境的任何其它信息。比如，想表达MDP的状态转移过程，可以将随机变量 $R$ 求积分得到 **状态转移概率**

$$
P_{ss^\prime}^a = p(s^\prime \vert s,a) \doteq \mathbb{P}\{ S_t=s^\prime \vert S_{t-1}=s, A_{t-1}=a \} = \sum_{r\in R}p(s^\prime,r \vert s,a)
$$

所以，整个马尔可夫决策过程的全部信息包含在状态变量集合 $A$，$S$，$R$ 和函数空间 $P$ 中，每个时刻都有一个 $A_t$，$S_t$，$R_t$，每两个相邻时刻之间都有一个 $p_t$。

类似地，状态动作 $s,a$ 对的期望奖励可以写作两个参数的函数

$$
r(s,a) = \mathbb{E}_\pi [R_{t+1} \vert S_t = s] = \sum_{r\in R} r \sum_{s^\prime \in S} p(s^\prime, r \vert s,a)
$$

### 4.4. 价值函数（Value Function）

马尔可夫决策过程中，价值函数分为状态价值函数和动作价值函数。

#### 4.4.1. 状态价值函数

**状态价值函数（state-value Function）是从某个状态 $s$ 开始，==执行策略 $\pi$== 所获得的回报的期望**；也即在执行当前策略时，衡量智能体处在状态 $s$ 时的价值大小。即

$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \vert S_t=s]
$$

其中，$\mathbb{E}_\pi[\cdot]$ 指在给定策略 $\pi$ 时一个随机变量的期望值，$t$ 可以是任意时刻。注意，终止状态的价值始终为零。我们把函数 $v_\pi(s)$ 称为策略 $\pi$ 的状态价值函数。

与 MRP 中的价值函数类似，状态价值函数也有如下贝尔曼方程成立

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(s^\prime) \vert S_t = s]\\
&=\sum_{a, s^\prime, r}\pi(a\vert s)p(s^\prime,r \vert s,a)\cdot [  r+\gamma v_\pi(s^\prime)  ]
\end{aligned}
$$

方程推导过程如下，首先易知

$$
\mathbb{E}_\pi[R_{t+1}\vert S_t=s] = \sum_a\pi(a\vert s)\sum_{s^\prime}\sum_r p(s^\prime, r \vert s,a) r
$$

而

$$
\begin{aligned}
\mathbb{E}_\pi[G_{t+1}\vert S_{t}=s] &= \sum_a \pi(a\vert s) \sum_{s^\prime}\sum_r  p(s^\prime,r \vert s,a) \mathbb{E}_\pi[G_{t+1}\vert S_{t+1}=s^\prime]\\
&= \sum_a \pi(a\vert s) \sum_{s^\prime}\sum_r  p(s^\prime,r \vert s,a) v_\pi(s^\prime)
\end{aligned}
$$

则有贝尔曼方程推导如下

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi[G_t \vert S_t=s]\\
&=\mathbb{E}_\pi[R_{t+1}+ \gamma G_{t+1}\vert S_t=s]\\
&=\sum_a \pi(a\vert s) \sum_{s^\prime}\sum_r p(s^\prime,r \vert s,a) [  r+\gamma v_\pi(s^\prime)  ] \\
&=\sum_{a, s^\prime, r}\pi(a\vert s)p(s^\prime,r \vert s,a)\cdot [  r+\gamma v_\pi(s^\prime)  ]
\end{aligned}
$$

最后一行，通过将求和符号合并后，我们可以看出，上述状等式描述了一个关于三参数 $a\in A, s^\prime \in S, r\in R$ 在所有可能性上的求和。对于每一个三元组，我们计算出其概率 $\pi(a\vert s)p(s^\prime,r \vert s,a)$ 然后乘以方括号内的值作为权值，最后甲醛加权求和得到状态价值函数的期望。

#### 4.4.2. 动作价值函数

类似地，我们把策略 $\pi$ 下在状态 $s$ 时采取动作 $a$ 的价值即为 $q_\pi(s,a)$。即根据策略 $\pi$，从状态 $s$ 开始，执行动作 $a$ 之后，所有可能的决策序列的期望回报

$$
q_\pi(s,a) = \mathbb{E}_\pi[G_t \vert S_t=s, A_t=a]
$$

与 MRP 中的价值函数类似，动作价值函数也有如下贝尔曼方程成立

$$
q_\pi(s,a) = \mathbb{E}_\pi[R_{t+1} + \gamma q_\pi(s^\prime,a^\prime) \vert S_t = s, A_t = a]
$$

#### 4.4.3. 回溯图与回溯操作

回溯操作就是将后继状态（或“状态-动作”二元组）的价值信息 _回传_ 给当前时刻的状态（或”状态-动作“二元组），可以用回溯图来表示，这是强化学习的核心内容。

典型的回溯图如下

![回溯图](/assets/img/postsimg/20221109/4-backup-diagram.png)

严谨地说，$q_\pi$ 和 $v_\pi$ 的作用是评估给定策略 $\pi$ 的价值，也就是一直使用这个策略来选取动作能得到的期望回报。不同之处是，$v_\pi$ 评估的对象是状态，考虑从状态 $s$ 出发，遵循策略 $\pi$ 得到的期望回报；$q_\pi$ 评估的对象是一个状态-动作对，考虑从状态 $s$ 出发，执行动作 $a$ 之后，遵循策略 $\pi$ 得到的期望回报。

因此，$v_\pi$ 可以写成 $q_\pi$ 关于策略 $\pi$（执行不同动作）的期望，$q_\pi$ 可以写成 $v_\pi$ 关于状态转移 $P_{ss^\prime}^a=p(s^\prime \vert s,a)$（执行动作 $a$ 后转移到不同状态）的期望。然后它们相互套娃，就得到了下面的两条等式，这两个等式也可以通过回溯图来直观理解。

[ **等式1** ]：

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_aq_\pi(s,a)\\
&= \sum_{a\in A}\pi(a\vert s)q_\pi(s,a)
\end{aligned}
$$

上面回溯图的上半部分对应上述等式，描述了处于特定状态 $s$ 的价值。即在状态 $s$ 时，遵循策略 $\pi$ 后，状态 $s$ 的价值体表示为在该状态下采取所有可能动作的动作价值（$q$ 值）按该状态下动作发生概率（策略 $\pi$）的乘积求和。

从状态 $s$ 来看，我们有可能采取两种行动（图中黑点），每个动作都有一个 $q$ 值（状态-动作值函数）。对 $q$ 值进行平均，这个均值告诉我们在特定状态下有多好，也即 $v_\pi(s)$。

上述等式可以通过 $v_\pi(s)$ 的贝尔曼方程推导得到，即

$$
\begin{aligned}
v_\pi(s) &= \sum_{a, s^\prime, r}\pi(a\vert s)p(s^\prime,r \vert s,a)\cdot [  r+\gamma v_\pi(s^\prime)  ]\\
&= \sum_{a}\pi(a\vert s)\cdot \sum_{s^\prime, r}p(s^\prime,r \vert s,a)\cdot [  r+\gamma v_\pi(s^\prime)  ]\\
&=\sum_{a}\pi(a\vert s)\cdot \sum_{s^\prime, r}p(s^\prime,r \vert s,a)\cdot [r+\gamma r+\gamma^2 r+...\vert s,a]\\
&=\sum_{a}\pi(a\vert s)\cdot \mathbb{E}_\pi[G_t\vert s,a]\\
&=\sum_{a}\pi(a\vert s)\cdot q_\pi(s,a)
\end{aligned}
$$

[ **等式2** ]：

$$
q_\pi(s,a) = R_s^a + \gamma\sum_{s\in S} p(s^\prime \vert s,a) v_\pi(s^\prime)
$$

证明如下

$$
\begin{aligned}
q_\pi(s,a) &= \mathbb{E}_\pi\left[ G_t \vert S_t=s, A_t=a \right]\\
&=\mathbb{E}_\pi\left[R_{t+1}+\gamma G_{t+1}\vert S_t=s, A_t=a \right]\\
&=\sum_{s^\prime,r}p(s^\prime,r \vert s,a) \left[r+\gamma \sum_{a^\prime}   \pi(a^\prime \vert s^\prime) \mathbb{E}_\pi [G_{t+1} \vert S_{t+1}=s^\prime, A_{t+1}=a^\prime]  \right]\\
&=\sum_{s^\prime,r}p(s^\prime,r \vert s,a) \left[r+\gamma \sum_{a^\prime}   \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime)  \right]\\
&=\sum_{s^\prime,r}p(s^\prime,r \vert s,a) \left[r+\gamma v_\pi(s^\prime) \right]\\\end{aligned}
$$

...(TODO)

（其它参考：https://zhuanlan.zhihu.com/p/478709774 ）

## 5. 参考文献

[1] 知乎. [强化学习（Reinforcement Learning）](https://www.zhihu.com/topic/20039099/intro).

[2] ReEchooo. [强化学习知识要点与编程实践（1）——马尔可夫决策过程](https://blog.csdn.net/qq_41773233/article/details/114698902)

[3] ReEchooo. [强化学习笔记（2）——马尔可夫决策过程](https://blog.csdn.net/qq_41773233/article/details/114435113)

[4] Ping2021. [第二讲 马尔可夫决策过程](https://zhuanlan.zhihu.com/p/494755866)

[5] 木头人puppet. [强化学习：贝尔曼方程和最优性](https://www.jianshu.com/p/9878238a1c9e)

[6] koch. [强化学习-贝尔曼方程和贝尔曼最优方程的推导](https://zhuanlan.zhihu.com/p/505723322)
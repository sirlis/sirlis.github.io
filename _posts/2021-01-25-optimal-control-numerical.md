---
title: 深度学习基础（高斯过程）
date: 2021-01-25 09:07:49 +0800
categories: [Academic, Knowledge]
tags: [optimalcontrol]
math: true
---

本文介绍了最优控制的数值解法的基础知识，包括微分方程的数值解法。

<!--more-->

 ---
 
- [1. 泰勒公式](#1-泰勒公式)
- [2. ODE-IPV的数值解法](#2-ode-ipv的数值解法)
  - [2.1. 时间推进法](#21-时间推进法)
    - [2.1.1. 线性多步法](#211-线性多步法)
    - [2.1.2. 多段法（Runge-Kutta 法）](#212-多段法runge-kutta-法)
- [3. ODE-BPV的数值解法](#3-ode-bpv的数值解法)
  - [3.1. 配点法](#31-配点法)
  - [3.2. 正交配点法](#32-正交配点法)
- [4. 参考文献](#4-参考文献)

# 1. 泰勒公式

人们希望通过简单的函数来近似表示复杂的函数。多项式就是一类简单函数，只包含加法和乘法两种基本运算。

给定函数 $f(x)$ ，要找一个在指定点 $x$ 附近 $f(x)$ 与之近似的多项式

$$
P(x) = a_0 + a_1x + a_2x^2 + \cdots + a_nx^n
$$

假设 $f(x)$ 在 $x_0$ 处可导，于是按照定义有

$$
f(x) = f(x_0) + f^\prime(x_0)(x-x_0) + O(x-x_0)
$$

这表明在 $x_0$ 附近可以用一次多项式来近似表达 $f(x)$ ，而误差是高于一阶的无穷小量。从几何角度看，这就是用曲线过点 $x_0$ 的切线来近似曲线。

从微分学的角度看，这种近似的特点是在点 $x_0$ 处多项式的函数值和一阶导数值与原始函数 $f(x)$ 相等。但是，许多情况下这个逼近程度不够，需要提高多项式的近似精度。

我们希望利用一个关于 $(x-x_0)$ 的 $n$ 次多项式来提高逼近精度

$$
P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)^2 + \cdots + a_n(x-x_0)^n
$$

我们希望该多项式在 $x_0$ 处的函数值及其直到 $n$ 阶导数值都与 $f(x)$ 的相应值分别相等，即

$$
\begin{aligned}
P_n(x_0) &= a_1 = f(x_0)\\
P^\prime_n(x_0) &= a_1 = f^\prime(x_0)\\
P^{\prime\prime}_n(x_0) &= 2a_2 = f^{\prime\prime}(x_0)\\
\cdots\\
P^{(n)}_n(x_0) &= n!a_n = f^{(n)}(x_0)\\
\end{aligned}
$$

于是有

$$
\begin{aligned}
P_n(x) = &f(x_0) \\
& + f^\prime(x_0)(x-x_0)\\
& + \frac{1}{2}f^{\prime\prime}(x_0)(x-x_0)^2\\
& + \cdots\\
& + \frac{1}{n!}f^{(n)}(x_0)(x-x_0)^n
\end{aligned}
$$

称 $P_n(x)$ 为 $f(x)$ 在点 $x_0$ 处的 $n$ 阶泰勒多项式。

# 2. ODE-IPV的数值解法

[微分方程的初值问题（ODE-IVP）](https://en.wikipedia.org/wiki/Initial_value_problem)如下

$$
\left\{
\begin{array}{l}
  \dot{x}=f(x(t),t),\quad t\in[t_n,t_{n+1}]\\
  x(t_0)=x_0
\end{array}
\right.
$$

其中，$f$ 为 $x,t$ 的已知函数，$x_0$ 为给定的初值。在以下讨论中，假设函数 $f(x,t)$ 在区域 $t_0\leq t\leq T, \vert x\vert<\infty$ 内连续，并且关于 $x$ 满足 Lipschitz 条件，使得

$$
\vert f(x, t) - f(\overline x, t) \vert \leq L\vert x - \overline x \vert
$$

由常微分方程理论，在以上假设下，初值问题必定且唯一存在数值解 $x(t)$。但是实际求解仍会存在很多困难，到目前为止我们只能对少数几个特殊类型的方程求得精确解，很多实际问题中常常得不到初等函数表示的解，需要求数值解。

假设 $t_n$ 时刻的状态量取值为 $x(t_n) = x_n$，则下一时刻 $t_{n+1}$ 的状态量取值 $x(t_{n+1}) = x_{n+1}$ 可以通过对原始微分式进行积分求得

$$
x_{n+1} = x_n + \int_{t_n}^{t_{n+1}}f(x(s),s)ds
$$

解决上述问题有两种方法：时间推进法和配点法。

## 2.1. 时间推进法

Time-Marching，时间推进法，微分方程在每个时刻的解根据前面一个或多个时刻的解求得。时间步进法再次被分为两类：多步法（multiple-step）和多阶段法（multiple-stage）。

### 2.1.1. 线性多步法

又称为 [linear multiple-step method](https://en.wikipedia.org/wiki/Linear_multistep_method)，即 $t_{n+1}$ 时刻微分方程的解由 $t_{n-j},\cdots,t_n$ 时刻的解求得，$j$ 为步长。

最简单的多步法就是单步法，即 $j=1$，最长用的单步法为**欧拉法**（Euler Method），具备如下的形式。

$$
x_{n+1} = x_n + h_n[b f_n + (1-b)f_{n+1}]
$$

其中 $f_n=f[x(t_n),t_n]$，$b\in[0,1]$，$h_n$ 是步长。

当 $b=1$ 时，为对应前向欧拉法；$b=0$ 时，为对应后向欧拉法；$b=1/2$ 时，为对应改进的欧拉法。欧拉法也可以从一阶泰勒多项式变化得到。

当 $j>1$ 时，就是更加复杂的线性多步法。形如

$$
\begin{aligned}
&a_0x_n + \cdots + a_{j-1}x_{n+j-1} + a_jx_{n+j} =\\
&h(b_0f(x_n,t_n) + \cdots + b_{j-1}f(x_{n+j-1},t_{n+j-1})+b_jf(x_{n+j},t_{n+j}))
\end{aligned}
$$

其中 $a_j=1$。系数 $a_0,\cdots,a_{j-1}$ 和 $b_0,\cdots,b_j$ 的选取决定了多步法的具体形式，一般在逼近程度和计算简便性上进行权衡。更加普遍的情况下，其中绝大部分的系数都置为0。

如果 $b_j=0$ 则称为显式法，因为可以直接根据等式计算 $x_{n+j}$。如果$b_j\neq 0$ 则称为隐式法，因为 $x_{n+j}$ 依赖于 $f(x_{n+j},t_{n+j})$，需要通过迭代的方法来求解，比如采用牛顿迭代法。

有时候，采用显式多步法来 『预测』 $x_{n+j}$，然后用隐式来 『校正』它，这种方式称为 预测-校正法（predictor–corrector method）。

下面列举两种常用的线性多步法家族。

**Adams-Bashforth methods**，一种显式法，其中 $a_{j-1}=-1$ 而 $a_{j-2}=\cdots=a_0=0$，然后设计 $b_j$ 来使得方法具备 $j$ 阶精度（同时也使得算法具备唯一性）。

$j=1,2,3$ 步 Adams-Bashforth 方法如下：

$$
\begin{aligned}
  x_{n+1} &= x_n + hf(x_n, t_n)\quad (前向欧拉法)\\
  x_{n+2} &= x_{n+1} + h[\frac{3}{2}f(x_{n+1}, t_{n+1})-\frac{1}{2}f(x_{n}, t_{n})]\\
  x_{n+3} &= x_{n+2} + h[\frac{23}{12}f(x_{n+2}, t_{n+2})-\frac{16}{12}f(x_{n+1}, t_{n+1})+\frac{5}{12}f(x_{n}, t_{n})]\\
\end{aligned}
$$

如何确定参数 $b+j$ 的方法略，可参考维基百科。

**Adams-Moulton methods**，一种隐式法，与 Adams-Bashforth 方法很类似，只是设计 $b_j$ 使得精度阶数尽可能高（$j$ 阶 Adams-Moulton 法具备 $j+1$ 阶精度，而$j$ 阶 Adams-Bashforth 法只具备 $j$ 阶精度）。

$j=1,2$ 步 Adams-Moulton 方法如下：

$$
\begin{aligned}
  x_{n+1} &= x_n + hf(x_{n+1}, t_{n+1})\quad (后向欧拉法)\\
  x_{n+1} &= x_n + \frac{1}{2}h[f(x_{n+1}, t_{n+1})+f(x_n, t_n)]\quad (梯形法则)\\
  x_{n+2} &= x_{n+1} + h[\frac{5}{12}f(x_{n+2}, t_{n+2})+\frac{3}{2}f(x_{n+1}, t_{n+1})-\frac{1}{12}f(x_{n}, t_{n})]\\
\end{aligned}
$$

### 2.1.2. 多段法（Runge-Kutta 法）

又称为 multiple-stage method，是在 $[t_n,t_{n+1}]$ 区间内划分若干临时段，然后进行迭代求解的一种常微分方程数值解法。其中最长用的是 **龙格库塔**（**Runge-Kutta**） 法。

定义步长为 $h$，将区间划分为 $s$ 个子区间，则 $s$ 阶显式Runge-Kutta 法为

$$
\begin{aligned}
x_{n+1} &= x_n + h\sum_{i=1}^sb_ik_i\\
k_1&= f(x_n,t_n)\\
k_2&= f(x_n+h(a_{21}k_1),t_n+c_2h)\\
k_3&= f(x_n+h(a_{31}k_1+a_{32}k_2),t_n+c_3h)\\
\vdots\\
k_s&= f(x_n+h(a_{s1}k_1+a_{s2}k_2+\cdots+a_{s,s-1}k_{s-1}),t_n+c_sh)\\
\end{aligned}
$$

根据泰勒展开，Runge-Kutta 当且仅当满足如下条件时才具备自洽性

$$
\sum_{i=1}^s b_i=1
$$

如果进一步要求方法具备 $p$ 阶精度，则需要补充相应的条件。比如一个 $s$ 阶 $p$ 级 Runge-Kutta 法需要满足 $s\geq p$ 且 $s\geq p+1(p\geq 5)$。一种常用的确定系数的条件为

$$
\sum_{j=1}^{i-1}a_{ij} = c_i,\ i=1,2,\cdots,s
$$

但是这个条件单独而言即非自洽性的必要条件也非充分条件。

$s=1$ 阶龙格库塔法就是显式欧拉法。

$s=4$ 阶龙格库塔法如下

$$
\begin{aligned}
x_{n+1} &=x_n + \frac{1}{6}h(k_1+2k_2+2k_3+k_4),\\
t_{n=1} &= t_n + h,\\
k_1 &= f(x_n,t_n),\\
k_2 &= f(x_n+h\frac{k_1}{2},t_n + \frac{h}{2}),\\
k_3 &= f(x_n+h\frac{k_2}{2},t_n + \frac{h}{2}),\\
k_4 &= f(x_n+hk_3,t_n + h).\\
\end{aligned}
$$

Runge-Kutta 公式的思路就是利用区间内一些特殊点的一阶导数值的线性组合来替代某点处的n阶导数值,这样就可以仅通过一系列一阶导数值来得到某点幂级数展开的预测效果。这和泰勒公式正好是反过来的,泰勒公式是用某点的n阶幂级数展开来近似得到小领域内的函数值。

# 3. ODE-BPV的数值解法

[微分方程的边值问题（ODE-BPV）](https://en.wikipedia.org/wiki/Boundary_value_problem) 类似初值问题。边值问题的条件是在区域的边界上，而初值问题的条件都是在独立变量及其导数在某一特定值时的数值（一般是定义域的下限，所以称为初值问题）。

例如独立变量是时间，定义域为 $[0,1]$，边值问题的条件会是 $x(t)$ 在 $t=0$ 及 $t=1$ 时的数值，而初值问题的条件会是 $t=0$ 时的 $x(t)$ 及 $x^\prime(t)$ 之值。

解决边值问题一般采用：

- [打靶法](https://en.wikipedia.org/wiki/Shooting_method)
- [差分](https://en.wikipedia.org/wiki/Finite_difference)
- [伽辽金法](https://en.wikipedia.org/wiki/Galerkin_method)
- [配点法](https://en.wikipedia.org/wiki/Collocation_method)

## 3.1. 配点法

配点法选择的有限维候选解空间（通常是展开到一定阶数的多项式）和域中的多个点（称为配点），并选择在配点处满足给定方程的解 。

一种简单的方式是采用如下形式的 n 阶分段多项式来近似解 $x$

$$
p(t) = \sum_{i=0}^n a_i(t-t_0)^i,\ t\in[t_0,t_1]
$$

假设希望求微分方程初值问题

$$
\left\{
\begin{array}{l}
  \dot{x}=f(x(t),t)\\
  x(t_0)=x_0
\end{array}
\right.
$$

在区间 $[t_0,t_0+c_kh]$ 的解，其中 $0<c_1<\cdots<c_n\leq1$。

使多项式满足以下两个约束

- 初始条件 $p(t_{0})=x_{0}$ 
- 微分方程 $\dot p(t_{k})=f(p(t_{k}),t_{k}),\ k=1,\cdots,n$

后者被称为配点条件，使得多项式在区间的每个临时点 $t_1,\cdots,t_n$ 上的微分均等于微分方程的等式右边。

上面两个约束提供了 $n+1$ 个条件，正好对应 $n$ 阶多项式中的 $n+1$ 个待定参数。

**举例**：梯形法/改进的欧拉法（两个配点 $c_1=0, c_2 = 1$，那么 $n=2$）

配点条件为

$$
\begin{aligned}
  p(t_0) &= x_0\\
  \dot p(t_0) &= f(p(t_0),t_0)\\
  \dot p(t_0+h) &= f(p(t_0+h),t_0+h)
\end{aligned}
$$

因为有三个配点条件，因此多项式的阶数为 2。假设多项式为如下形式

$$
p(t) = a_2(t-t_0)^2 + a_1(t-t_0) + a_0
$$

将多项式及其导数带入上面的配点条件，可以解出三个系数

$$
\begin{aligned}
a_0 &= x_0\\
a_1 &= f(p(t_0),t_0)\\
a_2 &= \frac{1}{2h}[f(p(t_0+h),t_0+h)-f(p(t_0),t_0)]\\
\end{aligned}
$$

则 $t_0+h$ 位置的微分方程的近似解为

$$
x_1 = p(t_0+h) = x_0 + \frac{1}{2}h[f(x_1,t_0+h)+f(x_0,t_0)]
$$

配点法包括三个种类：

- Gauss 配点法（始末端点 $t_k,t_{k+1}$ 均不是配点）
- Radau 配点法（始末端点 $t_k,t_{k+1}$ 任意一个是配点）
- Lobatto 配点法（始末端点 $t_k,t_{k+1}$ 均是配点）

所有这些配点法本质上都是隐式龙格库塔法，但不是所有龙格库塔法都是配点法。

从另外一个角度，龙格库塔法（包括一阶欧拉法）既可以看作是分段法，又可以看作是配点法。其中的区别在于，从配点法的形式来看，所有微分方程是同时被解出的（多项式的所有参数同时确定），而分段法中所有参数是迭代解出的。

类似地，配点法被认为是一种隐式解法，因为所有时刻的状态均同时被解出（所有多项式参数同时确定后，将所有时刻带入多项式得到所有时刻的状态量），有别于时间推进方法中状态量序列的一步步显式解出。

最后，配点法也不需要采用「预测」-「校正」策略。

## 3.2. 正交配点法

[orthogonal collocation methods](https://en.wikipedia.org/wiki/Orthogonal_collocation)，配点法中的一个非常常用的具体方法族。与一般配点法的不同在于其采用**正交多项式**。

具体而言，在正交配置方法中，配置点是某个正交多项式的根，一般为 [切比雪夫（Chebyshev）多项式](https://en.wikipedia.org/wiki/Chebyshev_polynomials) 或者 Legendre 多项式。

> **第一类切比雪夫多项式** $T_n$ 由以下递推关系确定：
$$
\begin{aligned}
  T_0(x) &= 1\\
  T_1(x) &= x\\
  T_{n+1}(x) &= 2xT_n(x) - T_{n-1}(x),\ n=1,2,\cdots\\
\end{aligned}
$$
前 4 阶第一类切比雪夫多项式为
$$
\begin{aligned}
  T_0(x) &= 1\\
  T_1(x) &= x\\
  T_2(x) &= 2x^2 - 1\\
  T_3(x) &= 4x^3 - 3x\\
  T_4(x) &= 8x^4 - 8x^2+1\\
\end{aligned}
$$
![](../assets/img/postsimg/20210125/01.350px-Chebyshev_polynomial.gif)
**第一类切比雪夫多项式的根**又被称为[切比雪夫节点](https://en.wikipedia.org/wiki/Chebyshev_nodes)，在 $[0,1]$ 区间内为
$$
x_k=cos(\frac{2k-1}{2n}\pi),\ k=1,\cdots,n
$$
形象的看，切比雪夫节点等价于 $n$ 等分单位半球的点的 $x$ 坐标（下图中 $n=10$）。
![](../assets/img/postsimg/20210125/02.Chebyshev-nodes-by-projection.svg)
对于任意区间 $[a,b]$，切比雪夫节点为
$$
x_k=\frac{1}{2}(a+b) + \frac{1}{2}(b-a)cos(\frac{2k-1}{2n}\pi),\ k=1,\cdots,n
$$
切比雪夫节点广泛用于多项式插值，因为他们具备一个很好的性质，即具有最小的龙格现象（插值边缘剧烈抖动发散）。

该方法特别适合求解非线性问题，


**举例**：[**高斯-勒让德配点法**（Gauss–Legendre methods）](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_method)

# 4. 参考文献

无。
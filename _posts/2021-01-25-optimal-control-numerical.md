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
- [2. 常微分方程的数值解法](#2-常微分方程的数值解法)
  - [2.1. 时间推进法](#21-时间推进法)
    - [2.1.1. 线性多步法](#211-线性多步法)
    - [2.1.2. 多段法](#212-多段法)
- [3. 参考文献](#3-参考文献)

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

# 2. 常微分方程的数值解法

微分方程的初值问题如下（ODE-IVP）

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
\bm{x}_{n+1} = x_n + h_n[\theta f_n + (1-\theta)f_{n+1}]
$$

其中 $f_n=f[x(t_n),t_n]$，$\theta\in[0,1]$，$h_n$ 是步长。

当 $\theta=1$ 时，为对应前向欧拉法；$\theta=0$ 时，为对应后向欧拉法；$\theta=1/2$ 时，为对应改进的欧拉法。欧拉法也可以从一阶泰勒多项式变化得到。

当 $j>1$ 时，就是更加复杂的线性多步法。形如

$$
\begin{aligned}
x_{n+j} + a_{j-1}x_{n+j-1} + \cdots + a_0x_n =\\
h(b_jf(x_{n+j},t_{n+j})+b_{j-1}f(x_{n+j-1},t_{n+j-1})+\cdots+b_0f(x_n,t_n))
\end{aligned}
$$

### 2.1.2. 多段法

又称为 multiple-stage method，

# 3. 参考文献

无。
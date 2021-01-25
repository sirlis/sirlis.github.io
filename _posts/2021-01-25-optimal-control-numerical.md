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
 
- [1. 简介](#1-简介)
- [2. 下载与安装](#2-下载与安装)
- [3. 配置 R 开发环境](#3-配置-r-开发环境)
  - [3.1. 安装 R 语言](#31-安装-r-语言)
  - [3.2. 安装 LanguageServer](#32-安装-languageserver)
  - [3.3. 安装扩展](#33-安装扩展)
  - [测试 R 环境](#测试-r-环境)
- [4. 参考文献](#4-参考文献)

# 1. 泰勒公式

# 2. 常微分方程的数值解法

微分方程的初值问题如下（ODE-IVP）

$$
\left\{
\begin{array}{l}
  \dot{\bm x}=f(\bm x(t),t),\quad t\in[t_i,t_{i+1}]\\
  x(t_0)=x_0
\end{array}
\right.
$$

其中，$f$ 为 $\bm x,t$ 的已知函数，$\bm x_0$ 为给定的初值。在以下讨论中，假设函数 $f(\bm x,t)$ 在区域 $\bm t_0\leq t\leq T, \vert x\vert<\infty$ 内连续，并且关于 $x$ 满足 Lipschitz 条件，使得

$$
\vert f(\bm x, t) - f(\overline \bm x, t) \vert \leq L\vert \bm x - \overline \bm x \vert
$$

由常微分方程理论，在以上假设下，初值问题必定且唯一存在数值解 $\bm x(t)$。但是实际求解仍会存在很多困难，到目前为止我们只能对少数几个特殊类型的方程求得精确解，很多实际问题中常常得不到初等函数表示的解，需要求数值解。

解决上述问题有两种方法：时间推进法和配点法。

## 2.1. 时间推进法

Time-Marching，时间推进法，微分方程在每个时刻的解根据前面一个或多个时刻的解求得。时间步进法再次被分为两类：多步法（multiple-step）和多阶段法（multiple-stage）。

### 2.1.1. 多步法

又称为 multiple-step methods，即 $t_{k+1}$ 时刻微分方程的解由 $t_{k-j},\cdots,t_k$ 时刻的解求得，$j$ 为步长。

最简单的多步法就是单步法，即 $j=1$，最长用的单步法为**欧拉法**（Euler Method），具备如下的形式。

$$
\bm{x}_{k+1} = \bm k + h_k[\theta \bm f_k + (1-\theta)\bm f_{k+1}]
$$

### 2.1.2. 线性多步法

# 3. 参考文献

无。
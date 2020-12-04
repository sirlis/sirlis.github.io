---
title: 深度学习文章阅读（TS模糊系统）
date: 2020-12-02 16:39:19 +0800
categories: [Academic, Paper]
tags: [fuzzy]
math: true
---

本文介绍了 TS 型模糊系统，由 Takagi 和 Sugeno 两位学者在 1985 年提出，主要思想是将非线性系统用许多线段相近的表示出来，即将复杂的非线性问题转化为在不同小线段上的问题。

<!--more-->

---
- [1. TS 模糊系统](#1-ts-模糊系统)
  - [1.1. 推理过程](#11-推理过程)
  - [1.2. 特性](#12-特性)
  - [1.3. 辨识算法](#13-辨识算法)
- [2. TS 模糊控制](#2-ts-模糊控制)
- [3. 广义 TS 模糊系统](#3-广义-ts-模糊系统)
- [4. 应用](#4-应用)
  - [4.1. Trajectory Prediction](#41-trajectory-prediction)
  - [4.2. Fuzzy Control](#42-fuzzy-control)
  - [4.3. Fuzzy Neural Network](#43-fuzzy-neural-network)
- [5. 参考文献](#5-参考文献)


# 1. TS 模糊系统

> Tomohiro Takagi and Michio Sugeno. **Fuzzy Identification of Systems and Its Applications to Modeling and Control**[J]. Fuzzy Identification of Systems, 1993.

> A mathematical tool to build a fuzzy model of a system where fuzzy implications and reasoning are used is presented in this paper. The premise of an implication is the description of fuzzy subspace of inputs and its consequence is a linear input-output relation. The method of identification of a system using its input-output data is then shown. Two applications of the method to industrial processes are also discussed: a water cleaning process and a converter in a steel-making process.

TS 模糊模型是由多个线性系统对同一个非线性系统进行拟合，利用模糊算法进行输入变量的解构，通过模糊演算推理再去模糊化，生成数条代表每组输入与输出关系的方程。

假设模糊集为 $A$，隶属度函数为 $A(x)$，$x$ 属于某论域 $X$。“$x$ 属于 $A$ 且 $y$ 属于 $B$” 表达为

$$
\vert x\ is\ A\ and\ y\ is\ B \vert = A(x) \land B(y)
$$

对于离散系统模型，令 $R_i$ 表示模糊系统的第 $i$ 条规则，其一阶 TS 模糊系统典型的模糊蕴含条件（Implication）句为

$$
R_i:\quad if\quad f(x_1\ is\ A_1,\ ...,\ x_k\ is\ A_k)\quad then\quad y = g(x_1, ..., x_k)
$$

在实际应用中，$f$ 为 $and$ 连接符，$g$ 为线性函数，即

$$
R:\quad if\quad x_1\ is\ A_1\ and\ ...\ and\ x_k\ is\ A_k\quad then\quad y = p_0+p_1x_1+...+p_kx_k)
$$

## 1.1. 推理过程

假设有 3 个上述格式的蕴含条件 $R_i,\ i=1,...,3$，分别为

$$
\begin{aligned}
R_1:&\quad if\quad x_1\ is\ small_1\ and\ x_2\ is\ small_2 & \quad then \quad y=x_1+x_2\\
R_2:&\quad if\quad x_1\ is\ big_1\ & \quad then \quad y=2x_1\\
R_2:&\quad if\quad x_2\ is\ big_2\ & \quad then \quad y=3x_2
\end{aligned}
$$

前提（Premise）中涉及到的隶属度函数定义为

![premise](../assets/img/postsimg/20201202/1.jpg)

假设输入 $x_1=12, x_2=5$，那么三个前提下的结论（Consequence）为

$$
\begin{aligned}
y_1 &= x_1+x_2 = 17\\
y_2 &= 2x_1 = 24\\
y_3 &= 3x_2 = 15
\end{aligned}
$$

相应的三个真值（True Value）为

$$
\begin{aligned}
t_1 &= small_1(x_1)\land small_2(x_2) = 0.25\land 0.375 = 0.25\\
t_2 &= big_1(x_1) = 0.2\\
t_3 &= big_2(x_2) = 0.375
\end{aligned}
$$

那么最终 $y$ 的取值为（此处采用加权平均法）

$$
y = \frac{t_1y_1+t_2y_2+t_3y_3}{t_1+t_2+t_3} \approx 17.8
$$

用一张表格可以列写如下

![result](../assets/img/postsimg/20201202/2.jpg)

## 1.2. 特性

优点：

- 相比分段线性逼近，数学形式更紧凑，连接处比较平滑；
- 相比原始的非线性函数，更加简明，方便进一步处理；
- 模糊划分可以包含有意义的语义条件，方便的讲人类语言规则形式表达的先验知识融入到模型建立过程中（模糊逻辑的功效）；
- 万能逼近定律表明 TS 模糊系统能以任意精度逼近非线性模型，适用于广泛类型的非线性系统。

另一方面，TS 模糊系统存在以下问题

- 隶属度函数全部由直线组成，不具备自适应性
- 不能保证参数的最优性
- 模糊规则数目无法最佳确定，即无法预知模型的复杂程度

## 1.3. 辨识算法

需要确定以下三个部分

- $x_i, ..., x_k$，前提变量；
- $A_1,...,A_k$，隶属度函数的参数，简记为隶属度参数；
- $p_0, p_1,...,p_k$，结论中的参数。

注意，前提中的变量不需要全部出现。前两个部分的确定和变量如何划分到模糊子空间有关，最后一个部分与模糊子空间中如何描述输入输出关系有关。论文作者提出依次逐层考虑如何确定。

假设一个一般的系统表示如下

$$
\begin{aligned}
R_1:&\quad if\quad x_1\ is\ A_1^1,\ ...,\ x_k\ is\ A_k^1\\
&\quad then\quad y=p_0^1 + p_1^1\cdot x_1+...+p^1_k\cdot x_k\\
&\quad \vdots\\
R_n:&\quad if\quad x_1\ is\ A_1^n,\ ...,\ x_k\ is\ A_k^n\\
&\quad then\quad y=p_0^n + p_1^n\cdot x_1+...+p^n_k\cdot x_k\\
\end{aligned}
$$

那么输出为

$$
y = \frac{\sum_{i=1}^n (A_1^i(x_1)\land...\land A_k^i(x_k))\cdot(p_0^i+p_1^ix_1+...+p_k^ix_k)}{\sum_{i=1}^n (A_1^i(x_1)\land...\land A_k^i(x_k))}
$$

假设

$$
\beta_i = \frac{A_1^i(x_1)\land...\land A_k^i(x_k)}{\sum_{i=1}^n (A_1^i(x_1)\land...\land A_k^i(x_k))}
$$

那么

$$
y = \sum_{i=1}^n\beta_i(p_0^i+p_1^ix_1+...+p_k^ix_k)
$$

当给定一组输入输出数据 $x_{1j},...,x_{kj}\rightarrow y_j\ (j=1,...,m)$ 时，可以通过 least squares method 来确定参数 $p_0^i, p_1^i,...,p_k^i$。

# 2. TS 模糊控制

> T. Taniguchi; K. Tanaka; H. Ohtake; H.O. Wang. **Model construction, rule reduction, and robust compensation for generalized form of Takagi-Sugeno fuzzy systems**. IEEE Transactions on Fuzzy Systems ( Volume: 9, Issue: 4, Aug 2001).

在线性矩阵不等式（linear matrix inequality, LMI）设计框架下，基于 TS 模糊模型的非线性控制得以广泛应用。一般分为三个阶段：

- 第一阶段：对非线性被控对象的模糊建模
  - 利用输入输出数据进行模糊模型辨识
  - 或 基于分区非线性思想的模糊系统构建（模糊 IF-THEN 规则）
- 第二阶段：模糊控制规则推导，它反映了模糊模型的规则结构，它通过所谓的并行分布式补偿（PDC）实现
- 第三阶段：模糊控制器设计，即确定反馈增益。

> This paper presents a systematic procedure of fuzzy control system design that consists of fuzzy model construction, rule reduction, and robust compensation for nonlinear systems. 
 
本文提出了一种模糊控制系统设计的系统程序，该程序由模糊模型构建，规则约简和非线性系统的鲁棒补偿组成。

# 3. 广义 TS 模糊系统

将 TS 模糊系统进行规范化描述如下。

给定 $m$ 个输入向量 $x_1,...,x_m$，$n$ 条模糊规则为 $R_1,...,R_n$，第 $i$ 条模糊规则的模糊子集分别为 $A^i_1,...,A^i_m$（相应的隶属度函数为 $A^i_j(x_j)$），各个模糊规则的真值为 $G_1, ..., G_n$，各个模糊规则对应的结论为 $y_1,...,y_n$，最终输出为 $y$，那么采用加权平均法的 TS 模糊系统为

$$
\begin{aligned}
y &= \frac{\sum_{i=1}^n G_iy_i}{\sum_{i=1}^n G_i}\\
G_i &= \prod_{j=1}^m A^i_j(x_j)
\end{aligned}
$$

其中 $\prod$ 为模糊化算子，通常采取**取小** "$\land$" 或者 **代数积** "$\cdot$" 计算。

若隶属度函数采用高斯隶属度函数形式，则可得到具有 $m$ 输入单输出、模糊规则数为 $n$ 的广义 TS 模糊系统

$$
\begin{aligned}
y &= \frac{\sum_{i=1}^n G_iy_i}{\sum_{i=1}^n G_i}\\
G_i &= \prod_{j=1}^m A^i_j(x_j) = \prod_{j=1}^m exp{(-\left\vert\frac{x_j - b_j^i}{a_j^i}\right\vert)}
\end{aligned}
$$

广义 TS 模糊系统可以以任意精度逼近被控对象，而模型的参数可以通过参数辨识方法获得。

# 4. 应用

## 4.1. Trajectory Prediction
> Multi-agent Trajectory Prediction with Fuzzy Query Attention. NIPS 2020.

## 4.2. Fuzzy Control
> Robust ${L_1}$ Observer-Based Non-PDC Controller Design for Persistent Bounded Disturbed TS Fuzzy Systems

## 4.3. Fuzzy Neural Network

> Developing deep fuzzy network with Takagi Sugeno fuzzy inference system

# 5. 参考文献

无。
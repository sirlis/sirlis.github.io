---
title: 深度学习文章阅读（广义TS模糊系统）
date: 2020-12-08 10:48:19 +0800
categories: [Academic, Paper]
tags: [fuzzy]
math: true
---

本文介绍了 2001 年 Taniguchi 等提出的一种广义 TS 模糊系统的建模方法、规则规约和鲁棒补偿方法。

<!--more-->

---
- [1. 引言](#1-引言)
- [2. 传统 TS 模糊系统](#2-传统-ts-模糊系统)
- [3. 扩展 TS 模糊系统](#3-扩展-ts-模糊系统)
  - [规则约减](#规则约减)
- [4. 参考文献](#4-参考文献)

> T. Taniguchi; K. Tanaka; H. Ohtake; H.O. Wang. **Model construction, rule reduction, and robust compensation for generalized form of Takagi-Sugeno fuzzy systems**. IEEE Transactions on Fuzzy Systems ( Volume: 9, Issue: 4, Aug 2001).

# 1. 引言

在线性矩阵不等式（linear matrix inequality, LMI）设计框架下，基于 TS 模糊模型的非线性控制得以广泛应用。一般分为三个阶段：

- 第一阶段：对非线性被控对象的模糊建模
  - 利用输入输出数据进行模糊模型辨识（Takagi and Sugeno, 1993 等）
  - **或** 基于分区非线性思想的模糊系统构建（模糊 IF-THEN 规则）
- 第二阶段：模糊控制规则推导，它反映了模糊模型的规则结构，它通过所谓的并行分布式补偿（PDC）实现
- 第三阶段：模糊控制器设计，即确定反馈增益。

> This paper presents a systematic procedure of fuzzy control system design that consists of fuzzy model construction, rule reduction, and robust compensation for nonlinear systems. 
 
本文提出了一种模糊控制系统设计的系统程序，该程序由模糊模型构建，规则约简和非线性系统的鲁棒补偿组成。

# 2. 传统 TS 模糊系统

形式如下

$$
\begin{aligned}
  &rule\ i\quad (i=1,2,\cdots,r):\\
  &{\rm IF}\ z_1(t)\ is\ M_{1i}\ and\ \cdots\ and\ z_p(t)\ is\ M_{pi}\\
  &{\rm THEN}\ \dot \boldsymbol x(t) = \boldsymbol A_i \boldsymbol x(t) + \boldsymbol B_i \boldsymbol u(t)
\end{aligned}
$$

其中

- $r$ 是规则个数
- $M_{ji}$ 是第 $j$ 个输入分量的模糊集
- $\boldsymbol x\in \mathbb R^n$ 是状态量，$\boldsymbol u\in \mathbb R^m$ 是输入量
- $\boldsymbol A_i\in \mathbb R^{n\times n}, \boldsymbol B_i\in \mathbb R^{n\times m}$ 是系数矩阵
- $z_1(t),\cdots,z_p(t)$ 是已知的前提变量，可能为可测量的状态量的函数、外部干扰，和/或时间，用 $\boldsymbol z(t)$ 来表示所有

给定一对 $[\boldsymbol x(t),\boldsymbol u(t),\boldsymbol z(t)]$，采用重心法（加权平均法）可以得到模糊系统的最终输出：

$$
\dot \boldsymbol x(t) = \sum_{i=1}^r h_i(\boldsymbol z(t))(\boldsymbol A_i \boldsymbol x(t) + \boldsymbol B_i \boldsymbol u(t))$$

其中

$$
\begin{aligned}
  h_i(\boldsymbol z(t)) &= \frac{\omega_i(\boldsymbol z(t))}{\sum_{i=1}^r\omega_i(\boldsymbol z(t))}\\
  \omega_i(z(t)) &= \prod_{j=1}^p M_{ji}(z_j(t))
\end{aligned}
$$

$h_i(\boldsymbol z(t))$ 是每条规则的归一化权重，$M_{ji}(z_j(t)$ 是第 $i$ 条规则中第 $j$ 个分量的模糊集 $M_{ji}$ 的隶属度值。

# 3. 扩展 TS 模糊系统

考虑某一状态量的分量 $x_i$，非线性系统表示如下：

$$
\dot x_i(t) = \sum_{j=1}^n f_{ij}(\boldsymbol z(t))x_j(t) + \sum_{k=1}^m g_{ik}(\boldsymbol z(t))u_k(t)
$$

类似地：

- $x_1(t)\cdots x_n(t)$ 是状态量，$u_1(t)\cdots u_m(t)$ 是输入量
- $z_1(t),\cdots,z_n(t)$ 是已知的变量，可能为状态量的函数、外部变量，和/或时间
- $f_{ij}(\boldsymbol z(t)), g_{ik}(\boldsymbol z(t))$ 是关于 $\boldsymbol z(t)$ 矩阵

注意，上式中的 $j$ 是遍历所有状态量，$j=1,2,\cdots,n$，$k$ 是遍历所有输入量，$k=1,2,\cdots,m$。如果将上式想象成一个描述系统运动的动力学方程组，那么这是一个标准的状态方程。该方程刻画了每个状态量的一阶导与所有状态量和控制（输入）量的线性组合关系。

定义如下的新变量（表示系数 $f_{ij},g_{ik}$ 的最大最小值）

$$
\begin{aligned}
  a_{ij1} &\equiv \mathop{\rm max}\limits_{\boldsymbol z(t)} f_{ij}(\boldsymbol z(t))\\
  a_{ij2} &\equiv \mathop{\rm min}\limits_{\boldsymbol z(t)} f_{ij}(\boldsymbol z(t))\\
  b_{ik1} &\equiv \mathop{\rm max}\limits_{\boldsymbol z(t)} g_{ik}(\boldsymbol z(t))\\
  b_{ik2} &\equiv \mathop{\rm min}\limits_{\boldsymbol z(t)} g_{ik}(\boldsymbol z(t))\\
\end{aligned}
$$

借助上述新定义的变量，可以将 $f_{ij}(\boldsymbol z(t)), g_{ik}(\boldsymbol z(t))$ 转化为用其最大最小值表达的形式（transforming into fuzzy model representation）：

$$
\begin{aligned}
  f_{ij}(\boldsymbol z(t)) &= h_{ij1}(\boldsymbol z(t))a_{ij1} + h_{ij2}(\boldsymbol z(t))a_{ij2}\\
  g_{ik}(\boldsymbol z(t)) &= v_{ik1}(\boldsymbol z(t))b_{ik1} + v_{ik2}(\boldsymbol z(t))b_{ik2}
\end{aligned}
$$

上式的权重参数满足

$$
\sum_{l=1}^2 h_{ijl}(\boldsymbol z(t)) = 1\quad \sum_{l=1}^2 v_{ikl}(\boldsymbol z(t)) = 1
$$

权重参数实质上就是隶属度函数的形式，可以定义如下

$$
\begin{aligned}
h_{ij1}(\boldsymbol z(t)) &= \frac{f_{ij}(\boldsymbol z(t))-a_{ij2}}{a_{ij1}-a_{ij2}}\\
h_{ij2}(\boldsymbol z(t)) &= \frac{a_{ij1} - f_{ij}(\boldsymbol z(t))}{a_{ij1}-a_{ij2}}\\
v_{ik1}(\boldsymbol z(t)) &= \frac{g_{ik}(\boldsymbol z(t))-b_{ij2}}{b_{ij1}-b_{ij2}}\\
v_{ik2}(\boldsymbol z(t)) &= \frac{b_{ik1} - g_{ik}(\boldsymbol z(t))}{b_{ik1}-b_{ik2}}\\
\end{aligned}
$$

最终可以将原始非线性系统表示为：

$$
\begin{aligned}
\dot x_i(t) &= \sum_{j=1}^n f_{ij}(\boldsymbol z(t))x_j(t) + \sum_{k=1}^m g_{ik}(\boldsymbol z(t))u_k(t)\\
&=\sum_{j=1}^n\sum_{l=1}^2 h_{ijl}(\boldsymbol z(t))a_{ijl}x_j(t) + \sum_{k=1}^m\sum_{l=1}^2 v_{ijl}(\boldsymbol z(t))b_{ikl}u_k(t)
\end{aligned}
$$

$i$ 是输入向量的维度，$j$ **也是**输入向量的维度（没想到吧~？表示每个输入向量的一阶导与所有输入向量的关系），$l$ 是取大取小值的维度。

将上述式子转为矩阵形式，如下

$$
\begin{aligned}
\dot \boldsymbol x(t) &=\sum_{i=1}^n\sum_{j=1}^n\sum_{l=1}^2 h_{ijl}(\boldsymbol z(t))a_{ijl} \boldsymbol U^A_{ij} \boldsymbol x(t) + \sum_{i=1}^n\sum_{k=1}^m\sum_{l=1}^2 v_{ijl}(\boldsymbol z(t))b_{ikl}\boldsymbol U^B_{ik}\boldsymbol u(t)\\
&=\sum_{i=1}^n\sum_{j=1}^n\sum_{l=1}^2 h_{ijl}(\boldsymbol z(t)) \boldsymbol A_{ijl} \boldsymbol x(t) + \sum_{i=1}^n\sum_{k=1}^m\sum_{l=1}^2 v_{ijl}(\boldsymbol z(t))\boldsymbol B_{ikl}\boldsymbol u(t)
\end{aligned}
$$

其中

$$
\begin{aligned}
\boldsymbol A_{ijl} &= \begin{bmatrix}
  0&\cdots&0&0&0&\cdots&0\\
  \vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\
  0&\cdots&0&0&0&\cdots&0\\
  0&\cdots&0&a_{ijl}&0&\cdots&0\\
  0&\cdots&0&0&0&\cdots&0\\
  \vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\
  0&\cdots&0&0&0&\cdots&0\\
\end{bmatrix}\\
\boldsymbol B_{ikl} &= \begin{bmatrix}
  0&\cdots&0&0&0&\cdots&0\\
  \vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\
  0&\cdots&0&0&0&\cdots&0\\
  0&\cdots&0&b_{ijl}&0&\cdots&0\\
  0&\cdots&0&0&0&\cdots&0\\
  \vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\
  0&\cdots&0&0&0&\cdots&0\\
\end{bmatrix}
\end{aligned}
$$

个人理解：上述两个矩阵就是为了前面的矩阵求和式而人工构造的，可以与后面的 $\boldsymbol x(t), \boldsymbol u(t)$ 做矩阵乘法，取到对应的元素乘积。也就是说，上述两个矩阵分别有 $n,m$ 个，且每个都是上面这种只有一个元素不为 0 的稀疏形式。

作者表明，$a_{ijl}, b_{ikl}$ 再规则约减中非常重要，上面矩阵和的式子在规则约减中十分方便。

## 规则约减

规则约减与使用LMI进行控制器设计的计算工作量密切相关。

基本思路：将非线性项 $f_{ij}(\boldsymbol z(t)), g_{ik}(\boldsymbol z(t))$

# 4. 参考文献

无。
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



# 4. 参考文献

无。
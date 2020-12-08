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
- [2. 参考文献](#2-参考文献)

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

# 2. 参考文献

无。
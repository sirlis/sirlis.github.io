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
    - [1.3.1. 结论参数辨识](#131-结论参数辨识)
    - [1.3.2. 前提参数辨识](#132-前提参数辨识)
    - [1.3.3. 前提变量的选择](#133-前提变量的选择)
  - [1.4. 广义 TS 模糊系统](#14-广义-ts-模糊系统)
- [2. 广义 TS 模糊系统 2](#2-广义-ts-模糊系统-2)
- [4. Trajectory Prediction](#4-trajectory-prediction)
  - [4.1. 归纳偏置](#41-归纳偏置)
  - [4.2. 预测架构](#42-预测架构)
  - [4.3. 交互模块](#43-交互模块)
  - [4.4. 模糊查询注意力模块](#44-模糊查询注意力模块)
  - [4.5. 分析](#45-分析)
  - [4.6. 训练](#46-训练)
  - [4.7. 实验](#47-实验)
- [5. 其它](#5-其它)
- [6. 参考文献](#6-参考文献)


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
R_i:\quad IF\quad f(x_1\ is\ A_1,\ \cdots,\ x_k\ is\ A_k)\quad THEN\quad y = g(x_1, \cdots, x_k)
$$

在实际应用中，$f$ 为 $and$ 连接符，$g$ 为线性函数，即

$$
R:\quad IF\quad x_1\ is\ A_1\ and\ \cdots\ and\ x_k\ is\ A_k\quad THEN\quad y = p_0+p_1x_1+\cdots+p_kx_k)
$$

## 1.1. 推理过程

假设有 3 个上述格式的蕴含条件 $R_i,\ i=1,\cdots,3$，分别为

$$
\begin{aligned}
R_1:&\quad IF\quad x_1\ is\ small_1\ and\ x_2\ is\ small_2 & \quad THEN \quad y=x_1+x_2\\
R_2:&\quad IF\quad x_1\ is\ big_1\ & \quad THEN \quad y=2x_1\\
R_2:&\quad IF\quad x_2\ is\ big_2\ & \quad THEN \quad y=3x_2
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

- $x_i, \cdots, x_k$，前提变量；
- $A_1,\cdots,A_k$，隶属度函数的参数，简记为隶属度参数；
- $p_0, p_1,\cdots,p_k$，结论中的参数。

注意，前提中的变量不需要全部出现。前两个部分的确定和变量如何划分到模糊子空间有关，最后一个部分与模糊子空间中如何描述输入输出关系有关。论文作者提出依次逐层考虑如何确定。

### 1.3.1. 结论参数辨识

假设一个一般的系统（$n$ 条规则）表示如下

$$
\begin{aligned}
R_1:&\quad IF\quad x_1\ is\ A_1^1,\ \cdots,\ x_k\ is\ A_k^1\\
&\quad THEN\quad y=p_0^1 + p_1^1\cdot x_1+\cdots+p^1_k\cdot x_k\\
&\quad \vdots\\
R_n:&\quad IF\quad x_1\ is\ A_1^n,\ \cdots,\ x_k\ is\ A_k^n\\
&\quad THEN\quad y=p_0^n + p_1^n\cdot x_1+\cdots+p^n_k\cdot x_k\\
\end{aligned}
$$

那么输出为

$$
y = \frac{\sum_{i=1}^n (A_1^i(x_1)\land\cdots\land A_k^i(x_k))\cdot(p_0^i+p_1^ix_1+\cdots+p_k^ix_k)}{\sum_{i=1}^n (A_1^i(x_1)\land\cdots\land A_k^i(x_k))}
$$

假设

$$
\beta_i = \frac{A_1^i(x_1)\land\cdots\land A_k^i(x_k)}{\sum_{i=1}^n (A_1^i(x_1)\land\cdots\land A_k^i(x_k))}
$$

那么

$$
y = \sum_{i=1}^n\beta_i(p_0^i+p_1^ix_1+\cdots+p_k^ix_k)
$$

当给定一组输入输出数据 $x_{1j},\cdots,x_{kj}\rightarrow y_j\ (j=1,\cdots,m)$ 时，可以通过最小二乘法来确定参数 $p_0^i, p_1^i,\cdots,p_k^i$。

> 最小二乘法：在实验中获得了自变量与因变量的若干组对应数据，在使偏差平方和取最小值时，找出一个已知类型的函数（即确定关系式中的参数）的方法。

经过 TS 模糊系统的推理后得到输出的估计为

$$
\begin{aligned}
\hat y_1 &= \sum_{i=1}^n\beta_{i1}(p_0^i+p_1^ix_{11}+\cdots+p_k^ix_{k1})\\
\hat y_2 &= \sum_{i=1}^n\beta_{i2}(p_0^i+p_1^ix_{12}+\cdots+p_k^ix_{k2})\\
&\cdots\\
\hat y_m &= \sum_{i=1}^n\beta_{im}(p_0^i+p_1^ix_{1m}+\cdots+p_k^ix_{km})\\
\end{aligned}
$$

对于其中第 $j$ 个式子，展开如下

$$
\begin{aligned}
\hat y_j &= \sum_{i=1}^n\beta_{ij}(p_0^i+p_1^ix_{1j}+\cdots+p_k^ix_{kj})\\
&= (\beta_{1j}p_0^1+\cdots+\beta_{nj}p_0^n)+(\beta_{1j}p_1^1+\cdots+\beta_{nj}p_1^n)x_{11}+\cdots\\
&= [\beta_{1j},\cdots,\beta_{nj}][p_0^1,\cdots,p_0^n]^T+[\beta_{1j}x_{11},\cdots,\beta_{nj}x_{11}][p_1^1,\cdots,p_1^n]^T+\cdots\\
&=\begin{bmatrix}
  \beta_{1j}\cdots\beta_{nj},\quad \beta_{1j}x_{11}\cdots\beta_{nj}x_{11},\quad \cdots
\end{bmatrix}
\begin{bmatrix}
  p_0^1\\
  \vdots\\
  p_0^n\\
  \\
  p_1^1\\
  \vdots\\
  p_1^n\\
  \\
  \vdots
\end{bmatrix}
\end{aligned}
$$

其中

$$
\beta_{ij} = \frac{A_{i1}(x_{1j})\land\cdots\land  A_{ik}(x_{kj})}{\sum_j A_{i1}(x_{1j})\land\cdots\land A_{ik}(x_{kj})}
$$

将上式的 $j$ 在 $[1,m]$ 上展开，可写成矩阵形式。假设 $X\in \mathbb R^{m\times n(k+1)}$，$Y,\hat Y\in \mathbb R^{m}$，$P\in \mathbb R^{n(k+1)}$，则

$$
\begin{aligned}
X &= \begin{bmatrix}
\beta_{11}\cdots\beta_{n1},\ \beta_{11}x_{11}\cdots\beta_{n1}x_{11},\ \cdots,\ beta_{11}x_{k1}\cdots\beta_{n1}x_{k1}\\
\cdots\\
\beta_{1m}\cdots\beta_{nm},\ \beta_{11}x_{1m}\cdots\beta_{nm}x_{1m},\ \cdots,\ \beta_{1m}x_{km}\cdots\beta_{nm}x_{km}
\end{bmatrix}\\
Y &= [y_1,\cdots,y_m]^T\\
\hat Y &= [\hat y_1,\cdots,\hat y_m]^T\\
P&=[p_0^1\cdots p_0^n,\cdots p_1^1\cdots p_1^n,\cdots,p_k^1\cdots p_k^n]^T
\end{aligned}
$$

$m$ 表示样本个数（$X,Y$ 的行数），$n$ 表示规则个数，$n(k+1)$ 表示待估计的特征参数 $P$ 的个数。

用矩阵形式表达的推理过程变为

$$
\hat Y = XP
$$

损失函数定义为

$$
J(P) = \frac{1}{2}(\hat Y-Y)^T(\hat Y-Y)= \frac{1}{2}(XP-Y)^T(XP-Y)
$$

根据最小二乘法原理，将损失函数对待估计参数求导取 0，结果为（组内大神推导表示无误）

> Eureka机器学习读书笔记. [最小二乘法（least sqaure method）](https://zhuanlan.zhihu.com/p/38128785/)

$$
\begin{aligned}
&\frac{\partial}{\partial P}J(P)= X^T(XP-Y)=0\\
&\Rightarrow X^TXP=X^TY\Rightarrow P=(X^TX)^{-1}X^TY
\end{aligned}
$$

即得到最小二乘法的标准解析解

$$
P=(X^TX)^{-1}X^TY
$$

如果能够提供足够数量的无噪声样本数据，最小二乘法可以精确估计出原始问题的真实参数。

**如果数据有噪声**，则采用稳态卡尔曼滤波（原文 stable-state，现在一般用 steady-state）来估计 $P$。稳态卡尔曼滤波可以计算出线性代数方程中的参数，使得均方差最小。

假设 $X$ 矩阵的第 $i$ 行为 $x_i$，$Y$ 的第 $i$ 个元素为 $y_i$，那么 $P$ 可以通过下面的式子递归估计（**涉及卡尔曼滤波的知识，还没看，假设就能估计出来了**）

$$
\begin{aligned}
P_{i+1} &= P_i + S_{i+1}\cdot x_{i+1}\cdot(y_{i+1}-x_{i+1}\cdot P_i)\\
S_{i+1} &= S_i-\frac{S_i\cdot x_i+x_{i+1}\cdot P_i}{1+x_{i+1}\cdot S\cdot x_{i+1}^T},\quad i=0,1,\cdots,m-1\\
P &= P_m
\end{aligned}
$$

初值为

$$
\begin{aligned}
P_0 &= 0\\
S_0 &= \alpha\cdot I\quad(\alpha=big\ number)
\end{aligned}
$$

最后给出一个例子。假设系统为

![kf1](../assets/img/postsimg/20201202/12.jpg)

在将模型的前提固定为原始系统的前提的情况下，将噪声添加到数据中，可以从输入输出数据中识别出后果，如下所示。

![kf2](../assets/img/postsimg/20201202/13.jpg)

下图展示了包含噪声的输入输出数据，原始结论和辨识出的结论。

![kf3](../assets/img/postsimg/20201202/14.jpg)

### 1.3.2. 前提参数辨识

在本节中，我们说明如何确定前提中的模糊集，即在**前提变量已经选定**的情况下，如何将前提变量的空间划分为模糊子空间（包括确定规则个数和确定每个规则中的模糊集/隶属度函数参数），但是规则个数作者只是一笔带过。

![inputoutputdata](../assets/img/postsimg/20201202/15.jpg)

如上图所示，根据图中的输入输出数据来划分 $x$ 的模糊子空间，比如 `x is small` 或者 `x is big`。即可设计如下两个规则

$$
\begin{aligned}
IF\ x\ is\ small\ THEN\ y=a_1x+b_1\\
IF\ x\ is\ big\ THEN\ y=a_2x+b_2\\
\end{aligned}
$$

然后需要确定 `small` 和 `big` 的隶属度函数，以及结论中的 $a_1,a_2,b_1,b_2$。

问题转变为，找寻隶属度函数的最优参数，使得性能指标最优。步骤如下

- 固定模糊集参数，通过上一节的方法得到最优的结论参数估计；
- 找寻隶属度函数的最优参数使得性能指标最优的问题可简化为一个非线性规划问题。作者采用著名的  **complex method for the minimization**（著名到我居然不认识）求解。由于传统的 TS 模糊系统的隶属度函数是线性的，因此用两个参数（分别对应取值为 0 和 1 的隶属度值）就能确定。

例子：使用从假定系统中收集的带有噪声的输入输出数据进行的识别，噪声的标准差是输出值的 5% 。

注意，如果不存在噪音，我们可以识别与原始系统相同的所有前提参数。指出这个事实非常重要。如果不是这种情况，我们就不能与模糊系统描述语言一起主张识别算法的有效性。

假设原始系统描述如下

![kf1](../assets/img/postsimg/20201202/12.jpg)

结论和带噪音的输入输出数据如下图所示。

![kf4](../assets/img/postsimg/20201202/16.jpg)

所识别的前提参数如下。我们可以看到已经推导出几乎相同的参数。

![kf5](../assets/img/postsimg/20201202/17.jpg)

### 1.3.3. 前提变量的选择

上一节假设前提变量已经给定。但是如何确定前提中用到哪些变量？因为给定的输入量 $x$ 可以不全用在前提中。

本质上包括两个问题：

- 选择哪些变量：选择一个变量意味着它的空间要被划分；
- 划分出多少子空间；

两个问题是有组合关系的，所以一般而言没有理论方法解决（The whole problem is a combinatorial one. So in general there seems no theoretical approach available）。作者提出一种启发式搜索方法，包含以下步骤：

假设一个包含 $k$ 个输入和 1 个输出的模糊系统。

- **步骤 1**：只划分 $x_1$ 为 `big` 和 `small`，其它分量不划分，意味着只有 $x_1$ 出现在前提中，其它分量不出现。那么模型规则如下
  
  $$
  \begin{aligned}
    IF\ x_1\ is\ big_1\ THEN\ \cdots\\
    IF\ x_1\ is\ small_1\ THEN\ \cdots\\
  \end{aligned}
  $$

  称上述模型为 模型 1-1，类似的，只划分第 $i$ 个分量的情况称为 模型 1-$i$。这样可以得到 $k$ 个模型，每个模型包含两个模糊蕴含条件（规则）。

- **步骤 2**：对上述每一个模型，用前面所述的方法确定最优的前提参数和结论参数。挑出其中性能指标最低的模型，作为稳定状态（stable state）。
- **步骤 3**：从前面的稳定状态出发，比如 模型 1-$i$，对所有分量 $x_i - x_j$ 进行排列组合，每个分量划分为 2 个模糊子空间。特别地，$x_i - x_i$ 组合将 $x_i$ 划分为 4 个模糊子空间，比如 `small, medium small, medium big, big`。这样又得到 $k$ 个模型，称为 模型 2-j。再次挑出其中性能指标最小的一个模型。
- **步骤 4**：重复**步骤 3**，往里再次添加一个其它分量。当满足下列任一条件时搜索停止：
  - 性能指标小于预设值；
  - 模糊蕴含条件的个数大于预设值；

整个过程如图所示

![vs](../assets/img/postsimg/20201202/18.jpg)


## 1.4. 广义 TS 模糊系统

将 TS 模糊系统进行规范化描述如下。

给定 $m$ 个输入向量 $x_1,\cdots,x_m$，$n$ 条模糊规则为 $R_1,\cdots,R_n$，第 $i$ 条模糊规则的模糊子集分别为 $A^i_1,\cdots,A^i_m$（相应的隶属度函数为 $A^i_j(x_j)$），各个模糊规则的真值为 $G_1, \cdots, G_n$，各个模糊规则对应的结论为 $y_1,\cdots,y_n$，最终输出为 $y$，那么采用加权平均法的 TS 模糊系统为

$$
\begin{aligned}
y &= \frac{\sum_{i=1}^n G_iy_i}{\sum_{i=1}^n G_i}\\
G_i &= \prod_{j=1}^m A^i_j(x_j)
\end{aligned}
$$

其中 $\prod$ 为模糊化算子，通常采取**取小** "$\land$" 或者 **代数积** "$\cdot$" 计算。

若隶属度函数采用高斯隶属度函数形式，则可得到具有 $m$ 输入单输出、模糊规则数为 $n$ 的广义 TS 模糊系统（未能找到出处）

$$
\begin{aligned}
y &= \frac{\sum_{i=1}^n G_iy_i}{\sum_{i=1}^n G_i}\\
G_i &= \prod_{j=1}^m A^i_j(x_j) = \prod_{j=1}^m exp{(-\left\vert\frac{x_j - b_j^i}{a_j^i}\right\vert)}
\end{aligned}
$$

广义 TS 模糊系统可以以任意精度逼近被控对象，而模型的参数可以通过参数辨识方法获得。

# 2. 广义 TS 模糊系统 2

> T. Taniguchi; K. Tanaka; H. Ohtake; H.O. Wang. **Model construction, rule reduction, and robust compensation for generalized form of Takagi-Sugeno fuzzy systems**. IEEE Transactions on Fuzzy Systems ( Volume: 9, Issue: 4, Aug 2001).

在线性矩阵不等式（linear matrix inequality, LMI）设计框架下，基于 TS 模糊模型的非线性控制得以广泛应用。一般分为三个阶段：

- 第一阶段：对非线性被控对象的模糊建模
  - 利用输入输出数据进行模糊模型辨识（Takagi and Sugeno, 1993 等）
  - **或** 基于分区非线性思想的模糊系统构建（模糊 IF-THEN 规则）
- 第二阶段：模糊控制规则推导，它反映了模糊模型的规则结构，它通过所谓的并行分布式补偿（PDC）实现
- 第三阶段：模糊控制器设计，即确定反馈增益。

> This paper presents a systematic procedure of fuzzy control system design that consists of fuzzy model construction, rule reduction, and robust compensation for nonlinear systems. 
 
本文提出了一种模糊控制系统设计的系统程序，该程序由模糊模型构建，规则约简和非线性系统的鲁棒补偿组成。

# 4. Trajectory Prediction

> NIPS 2020. **Multi-agent Trajectory Prediction with Fuzzy Query Attention**.

做多目标轨迹预测。

## 4.1. 归纳偏置

inductive biases，归纳偏置。

> LinT. [如何理解Inductive bias？](https://www.zhihu.com/question/264264203/answer/830077823)
> 归纳偏置在机器学习中是一种很微妙的概念：在机器学习中，很多学习算法经常会对学习的问题做一些假设，这些假设就称为归纳偏置(Inductive Bias)。归纳偏置这个译名可能不能很好地帮助理解，不妨拆解开来看：归纳(Induction)是自然科学中常用的两大方法之一(归纳与演绎, induction and deduction)，指的是从一些例子中寻找共性、泛化，形成一个比较通用的规则的过程；偏置(Bias)是指我们对模型的偏好。因此，归纳偏置可以理解为，从现实生活中观察到的现象中归纳出一定的规则(heuristics)，然后对模型做一定的约束，从而可以起到“模型选择”的作用，即从假设空间中选择出更符合现实规则的模型。其实，贝叶斯学习中的“先验(Prior)”这个叫法，可能比“归纳偏置”更直观一些。
> 以神经网络为例，各式各样的网络结构/组件/机制往往就来源于归纳偏置。在卷积神经网络中，我们假设特征具有局部性(Locality)的特性，即当我们把相邻的一些特征放在一起，会更容易得到“解”；在循环神经网络中，我们假设每一时刻的计算依赖于历史计算结果；还有注意力机制，也是基于从人的直觉、生活经验归纳得到的规则。

- **惯性**（Inertia）：几乎所有无生命实体都按照匀速前进，除非收到外力作用。这个规则在作为一阶近似估计时，在段时间内同样适用于有生命实体（如行人），因为行人几乎也以匀速行走，除非需要转弯或减速以避免碰撞；
- **运动的相对性**（Motion is relative）：两个目标之间的运动是相对的，在预测未来轨迹时应该使用他们之间的相对位置和速度（相对观测，relative observations），对未来的预测也需要是相对于当前位置的偏差（相对预测，relative predictions）；
- **意图**（Intent）：有生命对象有自己的意图，运动会偏离惯性，需要在预测模型中进行考虑；
- **交互**（Interactions）：有生命对象和无生命对象可能偏离它们预期的运动，比如受到其它附近对象的影响。这种影响需要清晰的建模。

## 4.2. 预测架构

下图 (a) 为预测架构，输入 $t$ 时刻的所有对象的位置 $p^t_{i=1:N}$。使用 $t\leq T_{obs}$ 时刻的位置作为观测，对 $t\geq T_{obs}$ 时刻的位置进行预测。我们对每个对象的下一时刻位置 $\hat p^{t+1}_i$ 进行预测，预测量是相对于当前时刻 $p_i^t$ 的位置偏差（relative prediction）。

![tp](../assets/img/postsimg/20201202/9.jpg) 

- **公式 1**：将位置偏差拆分为一阶常速度估计 $\tilde v_i^t$ （惯性）和速度修正项 $\Delta v_i^t$ （意图和对象间的交互）。
- **公式 2**：一阶常速度估计由当前时刻位置和前一时刻位置直接差分得到。
- **公式 3**：采用 LSTM 来捕捉每个对象的意图，其隐藏状态 $h_i^t$ 能够保存之前的轨迹信息。LSTM 的权重参数所有对象均共享。为了计算速度修正项 $\Delta v_i^t$，首先用 LSTM 根据每个对象的当前位置初步更新一个临时隐藏状态 $\tilde h_i^t$。
- **公式 4**：然后将对象当前位置和临时隐层状态 $h_i^t,$ 同时送入一个 「交互模块（Interaction module）」 来推理对象间的交互、合计他们的效果，然后更新每个对象的隐藏状态，同时计算对象的速度修正项。和所有对象的当前位置向量进一步被用于将来的对象间的交互，汇总其效果并更新每个对象的隐藏状态，

$$
\begin{aligned}
\hat p^{t+1}_i &= p_i^t + \tilde v_i^t + \Delta v_i^t,&\quad \forall i\in 1:N\\
(Inertia): \tilde v_i^t &= p_i^t - p_i^{t-1},&\quad \forall i\in 1:N\\
(Intents): \tilde h_i^t &= LSTM(p_i^t, h_i^{t-1}),&\quad \forall i\in 1:N\\
(Interactions): h_i^t,\Delta v_i^t&= InteractionModule(p_i^t, \tilde h_i^t)&
\end{aligned}
$$

由于所有的计算都在当前时刻 $t$ 下，因此后文可以略去该上标。

## 4.3. 交互模块

下图 (b) 作为交互模块。

![im](../assets/img/postsimg/20201202/10.jpg) 

- **公式 1**：在每两个对象间**产生一条有向边来建立一张图（creates a graph by generating directed edges between all pairs of agents）**（忽略自身与自身的边连接），得到边集合 $\varepsilon$。将边集合、所有对象的位置和临时隐藏状态（意图）估计送入 模糊查询注意力模块（Fuzzy Query Attention, FQA）得到每一个对象的注意力向量 $a_i$，该向量汇聚了其与其它所有对象的交互信息。
  <font color=red>有向边的方向如何确定的不明确。</font>

- **公式 2，3**：将注意力向量、位置向量、临时隐藏状态（意图）送入随后的 2 层全连接层（ReLU），得到更新后的隐藏状态 $h_i$（且之后返传给 LSTM 作为上一时刻的隐藏状态），再次经过 2 层全连接层（ReLU）得到每个对象的速度修正项 $\Delta v_i$。

$$
\begin{aligned}
a&=FQA(p,\hat h, \varepsilon)&\\
h_i&= FC_2(ReLU(FC_1(p_i, h_i, a_i))),&\quad \forall i\in 1:N\\
\Delta v_i&= FC_4(ReLU(FC_3(p_i))),&\quad \forall i\in 1:N
\end{aligned}
$$

## 4.4. 模糊查询注意力模块

![fqa](../assets/img/postsimg/20201202/11.jpg) 

FQA 模块将有向边图看作 发送-接收 对象对（sender-receiver pairs of agents）。从高层来看，该模块建模了所有发送对象对一个特定接收对象的汇聚影响。基本想法是建立 key-query-value self attention 网络（Vaswani et al）。

> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. **Attention Is All You Need**[J]. arXiv preprint arXiv:1706.03762v5 [cs.CL], 2017. [Google Transformer]
> 
> **Self attention**: 输入一系列向量，输出所有向量对每个特定向量的 value 的加权和（注意力 / 重要性）。value 是每个向量乘以一个权重矩阵得到的，矩阵需要被训练。

- **产生独立特征**：复制 $p,\hat h$ 到每一条边，一条边两个对象，一个 sender 一个 receiver，那么就复制出 $p_s, p_r, h_s, h_r$。
- **产生相对特征**：$p_{sr} = p_s-p_r$ （相对位移），$h_{sr} = h_s-h_r$ （相对状态），$\hat p_{sr} = p_{sr}/\vert\vert p_{sr} \vert\vert$（单位化）,$\hat h_{sr} = h_{sr}/\vert\vert h_{sr} \vert\vert$（单位化）。这些特征用来捕捉相对观测归纳偏差。
- 对于每一条边，将上述所有**特征拼接** $f_{sr} = \{ p_s, p_r, p_sr, \hat p_{sr}, h_s, h_r, h_sr, \hat h_{sr} \}$，然后分别经过一个 **单层全连接层**，产生 $n$ 个 keys $K_{sr}\in \mathbb R^{n\times d}$ 和  $n$ 个 queries $Q_{sr}\in \mathbb R^{n\times d}$。$n$ 代表 $s-r$ 对的个数（个人理解也就是边的个数，也就是 $f_{sr} 的个数$），$d$ 应该是用户定义的维度。
- 将 $K_{sr}\in \mathbb R^{n\times d}$ 和 $Q_{sr}\in \mathbb R^{n\times d}$ 通过一个**点乘的变体**（元素积然后按行求和，row-wise dot-product），产生模糊决策 $D_{sr}\in \mathbb R^n$。
- 
$$
\begin{aligned}
K_{sr}&=FC_5(f_{sr}^\perp)&\ \forall (s,r)\in 1:N,s\neq r\\
Q_{sr}&=FC_6(f_{sr}^\perp)&\ \forall (s,r)\in 1:N,s\neq r\\
D_{sr}&= \sigma(K_{sr}\star Q_{sr}+B)=\left( \sum_{dim=1} K_{sr}\odot Q_{sr}+B\right),&\ \forall (s,r)\in 1:N,s\neq r
\end{aligned}
$$

注：作者说的模糊决策不是模糊逻辑，而是浮点数取值的决策，相对于离散取值的布尔决策。「大意了！」

其中 $B$ 是一个待优化的偏差参数矩阵，$\sigma$ 是 sigmoid 激活函数，$\perp$ 是分离运算符（detach operator）。[不允许梯度从 $Q,K$ 回传，只能从 $V$ 回传]

> The detach operator acts as identity for the forward-pass but prevents any gradients from propagating back through its operand. This allows us to learn feature representations only using responses while the keys and queries make useful decisions from the learnt features.
> we learn the sender-receiver features by backpropagating only through the responses ($V_sr$) while features are detached to generate the keys and queries. This additionally allows us to inject human knowledge into the model via handcrafted non-learnable decisions, IF such decisions are available 

最终得到的模糊决策 $D_{sr} \in [0,1]^n$ 可以解释为一组 $n$个 连续取值的决策，反应了 sender 对象和 receiver 对象之间的交互。这个决策可以用来影响（select）receiver 对象对 sender 对象当前状态的应答。

- **确定性应答（公式 1，2）**：相对特征并行的通过两个 2 层全连接层（第一层包含 ReLU 激活函数）产生 yes-no 应答 $V_{y,sr},V_{n,sr}\in \mathbb R^{n\times d_v}$，与 $D_{sr}=1\ or\ D_{sr}=0$ 对应。虽然可以使用全部特征 $f_{sr}$，但实验表明只用一部分特征（$p_{sr},h_s$）的表现就很好了，还能节约参数。
- **模糊应答（公式 3）**：将上述确定性应答模糊化，根据模糊决策 $D_{sr}$ 和其补集 $\overline D_{sr}= 1 - D_{sr}$ 通过 fuzzy IF-else 产生最终的模糊应答。

$$
\begin{aligned}
V_{y,sr}&=FC_8(ReLU(FC_7(p_{sr},h_s))),&\quad \forall (s,r)\in 1:N,s\neq r\\
V_{n,sr}&=FC_{10}(ReLU(FC_9(p_{sr},h_s))),&\quad \forall (s,r)\in 1:N,s\neq r\\
V_{sr}&=D_{sr}V_{y,sr}+\overline D_{sr}V_{n,sr}&\quad \forall (s,r)\in 1:N,s\neq r\\
\end{aligned}
$$

最后得到 $n$ 个对象对的应答 $V_{sr}\in \mathbb R^{n\times d_v}$。

- **公式 1**：将应答拼接 $\in \mathbb R^{nd_v}$，然后过一个全连接层，提高向量维度增加信息量，以弥补后续最大池化带来的信息丢失。
- **公式 2**：对上述步骤的输出进行最大池化，将所有对 receiver 对象的交互的影响累积。
- **公式 3**：最后再通过一个全连接层降维（与之前升维对应）。

$$
\begin{aligned}
V_{proc,sr} &= FC_{11}(concat(V_{sr}))\\
V_{proc,r} &= maxpool_{s:(s-r)\in\varepsilon}V_{proc,sr}\\
a_r&=FC_{12}(V_{proc,r}),\quad \forall r\in 1:N
\end{aligned}
$$

## 4.5. 分析

上述架构受到 multi-head self-attention 的启发，但是经过了大量改造。

- 从 self-attention 改成 pairwise-attention；
- 包括一个可学习的 $B$ 使得模型能力更高；
- 从矩阵元素积变为元素积然后按行求和，降低计算量和硬件性能要求，同时保证了性能；
- 只允许梯度从 $V_{sr}$ 回传。这使得额外增加不可学习的人类知识成为可能（section 4.3）？

FQA 能学到：

- 靠近（Proximity）：假设 $K,Q$ 是 $p_{sr}$ 且对应的 $B$ 是 $-d_{th}^2$ 那么决策 $D = \sigma(p_{sr}^Tp_{sr}-d_{th}^2)$ 逼近 0 表示两个对象 $s$ 和 $r$ 间的距离小于 $d_{th}$。注意到上述决策依赖 $B$ 的存在，即 $B$ 赋予模型更灵活的能力；
- 接近（Approach）：由于部分隐藏状态内部能够学习如何对对象的速度进行建模，FQA 可能可以学习到一种 $K_{sr} = v_{sr},Q_{sr} = \hat p_{sr},B=0$ 形式，这种形式逼近 0 表示两个对象相互直接接近对方。虽然我们并没有直接要求 FQA 学习这些可解释的决策，但是实验表明 FQA 学习到的模糊决策能够高度预测对象间的交互（section 4.3）。

## 4.6. 训练

用 MSE，评估下一时刻所有对象的预测位置与真实位置的偏差，用 Adam，batch size = 32， 初始学习率 0.001，每 5 epoch 乘以 0.8 下降。所有待比较的模型都训练至少 50 epoch，然后当连续 10 epoch 的验证 MSE 不下降时激活 early stopping，最多进行 100 epochs 训练。

所有样本的 $T_{obs} = \frac{2T}{5}$，我们遵循动态时间表，允许所有模型查看 $T_{temp}$ 时间步长的真实观测值，然后预测 $T-Ttemp$ 时间步长。在起始阶段，$T_{temp} = T$，然后每次减 1 直到 $T_{temp} = T_{obs}$。发现这样操作可以提高所有模型的预测性能：

- 观察 $T$ 个步长，预测 $T-T=0$ 个步长；
- 观察 $T-1$ 个步长，预测 $T-(T-1)=1$ 个步长；
- 观察 $T-2$ 个步长，预测 $T-(T-2)=2$ 个步长；
- ......； 
- 观察 $T_{obs}$ 个步长，预测 $T-(T-T_{obs})=T_{obs}$ 个步长； 

## 4.7. 实验

采用以下几个前人研究的数据集（包含不同种类的交互特征）。如果数据集没有划分，那么我们按照 $70:15:15$ 来划分训练集、验证集和测试集。

- ETH-UCY：3400 场景，T = 20；
- Collisions：9500 场景，T = 25；
- NGsim：3500 场景，T = 20；
- Charges：3600 场景，T = 25；
- NBA：7500 场景，T = 30。

baselines：

- Vanilla LSTM：
- Social LSTM：
- GraphSAGE：
- Graph Networks：
- Neural Relational Inference：
- Graph Attention Networks：


# 5. 其它
> Robust ${L_1}$ Observer-Based Non-PDC Controller Design for Persistent Bounded Disturbed TS Fuzzy Systems



# 6. 参考文献

无。
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
- [2. 广义 TS 模糊系统](#2-广义-ts-模糊系统)
- [3. TS 深度模糊网络](#3-ts-深度模糊网络)
  - [3.1. 网络结构](#31-网络结构)
  - [3.2. 网络参数辨识](#32-网络参数辨识)
    - [3.2.1. 前向传播](#321-前向传播)
    - [3.2.2. 反向传播](#322-反向传播)
  - [> $Q$ denotes the total number of rules in which the corresponding MF appears in premise part.](#-q-denotes-the-total-number-of-rules-in-which-the-corresponding-mf-appears-in-premise-part)
  - [实验](#实验)
    - [准备工作](#准备工作)
- [4. TS 模糊控制](#4-ts-模糊控制)
- [5. Fuzzy Control](#5-fuzzy-control)
- [6. Trajectory Prediction](#6-trajectory-prediction)
- [7. 参考文献](#7-参考文献)


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
R_i:\quad if\quad f(x_1\ is\ A_1,\ \cdots,\ x_k\ is\ A_k)\quad then\quad y = g(x_1, \cdots, x_k)
$$

在实际应用中，$f$ 为 $and$ 连接符，$g$ 为线性函数，即

$$
R:\quad if\quad x_1\ is\ A_1\ and\ \cdots\ and\ x_k\ is\ A_k\quad then\quad y = p_0+p_1x_1+\cdots+p_kx_k)
$$

## 1.1. 推理过程

假设有 3 个上述格式的蕴含条件 $R_i,\ i=1,\cdots,3$，分别为

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

- $x_i, \cdots, x_k$，前提变量；
- $A_1,\cdots,A_k$，隶属度函数的参数，简记为隶属度参数；
- $p_0, p_1,\cdots,p_k$，结论中的参数。

注意，前提中的变量不需要全部出现。前两个部分的确定和变量如何划分到模糊子空间有关，最后一个部分与模糊子空间中如何描述输入输出关系有关。论文作者提出依次逐层考虑如何确定。

假设一个一般的系统表示如下

$$
\begin{aligned}
R_1:&\quad if\quad x_1\ is\ A_1^1,\ \cdots,\ x_k\ is\ A_k^1\\
&\quad then\quad y=p_0^1 + p_1^1\cdot x_1+\cdots+p^1_k\cdot x_k\\
&\quad \vdots\\
R_n:&\quad if\quad x_1\ is\ A_1^n,\ \cdots,\ x_k\ is\ A_k^n\\
&\quad then\quad y=p_0^n + p_1^n\cdot x_1+\cdots+p^n_k\cdot x_k\\
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

当给定一组输入输出数据 $x_{1j},\cdots,x_{kj}\rightarrow y_j\ (j=1,\cdots,m)$ 时，可以通过 least squares method 来确定参数 $p_0^i, p_1^i,\cdots,p_k^i$。

# 2. 广义 TS 模糊系统

将 TS 模糊系统进行规范化描述如下。

给定 $m$ 个输入向量 $x_1,\cdots,x_m$，$n$ 条模糊规则为 $R_1,\cdots,R_n$，第 $i$ 条模糊规则的模糊子集分别为 $A^i_1,\cdots,A^i_m$（相应的隶属度函数为 $A^i_j(x_j)$），各个模糊规则的真值为 $G_1, \cdots, G_n$，各个模糊规则对应的结论为 $y_1,\cdots,y_n$，最终输出为 $y$，那么采用加权平均法的 TS 模糊系统为

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

# 3. TS 深度模糊网络

> 2017 . Developing deep fuzzy network with Takagi Sugeno fuzzy inference system. IEEE Transactions on Fuzzy System

## 3.1. 网络结构

提出了一种新型的三层 **TS Deep Fuzzy Network (TSDFN)** 网络架构。

TSDFN 的网络架构如下图所示

![tsdfn](../assets/img/postsimg/20201202/3.jpg)

图中，隐层（hidden layer）中的每一个神经元都是一个 TSFIS ，输出层只有一个神经元，也是一个 TSFIS 。当然也可以扩展为多输出，不同的输出间相互独立。

> FIS：fuzzy inference system，模糊推理系统，是一个完整的输入-输出模糊系统，比如上面介绍的 TS 模糊系统，就被称为 TSFIS

一个 TSFIS 神经元的模糊规则基（Fuzzy Rul Base，FRB）包含多条模糊规则，每条规则都包括前提部分和结论部分。一阶 TSFIS 的结论是输入的线形方程。FRB 的规则形式如下

$$
\begin{aligned}
R_i^h:&\quad {\rm IF}\quad x_1\ is\ G_{1,i}\ {\rm AND}\ \cdots\ {\rm AND}\ x_D\ is\ G_{D,i}\quad\\
&\quad {\rm THEN}\quad y\ is\ y_i=p_{i,0}+p_{i,1}x_1+\cdots+p_{i,D}x_D
\end{aligned}
$$

$D$ 是输入个数，$x_d$ 是第 $d$ 个输入分量（$d=1,\cdots,D$）。$R$ 是规则总个数$G_{d,i}$ 是前提中相应的输入模糊隶属度函数（$i=1,\cdots,R$）。前提中采用 "AND" 作为模糊连接符。

一个 TSFIS 的参数即为输入前提模糊隶属度函数的参数和结论系数，二者的组合可表示特定输入的模糊结构。可采用多种模糊隶属度函数。采用不同的模糊连接符可以定义不同的模糊规则基。

整个网络包括如下参数：

- 模糊规则的前提（premise）中的输入隶属度的参数；
- 每一层的每一个 TS 模糊神经元的结论部分的输入系数；

一个 TS 模糊神经元（TSFN）建模出了一种输入的复杂函数，输入的隶属度函数代表了模糊区域，建模出了输入数据的不确定性。模糊区域可以表示语义标签。TSDFN 中的 TSFN 提取输入数据中的复杂模式，相应的FRB参数以模糊规则的形式表示模式的内部结构。

> a TSFN in TSDFN extracts a complex pattern in input data and corresponding FRB parameters represent the nternal structure of the pattern in the form of fuzzy rules.

## 3.2. 网络参数辨识

采用标准的误差反向传播来针对特定数据进行网络参数辨识。

### 3.2.1. 前向传播

下面考虑 **一个一般的隐层 TSFN**（$S_h$），假设输入向量为 $\boldsymbol x=[x_1,x_2,\cdots,x_d,\cdots,x_D]$。

> $θ^h_{d,f}$ denotes parameter of $f^th$ input MF of input $d$ in premise part of **a rule** in FRB of $S_h$

$\boldsymbol \theta^h$ 表示**某个**规则中的输入隶属度函数的参数矩阵，那么

$$
\begin{aligned}
\boldsymbol \theta^h = \begin{bmatrix}
\theta^h_{1,1} & \cdots & \theta^h_{1,f} & \cdots & \theta^h_{1,F}\\
\vdots & \ddots & \vdots & \ddots & \vdots\\ 
\theta^h_{d,1} & \cdots & \theta^h_{d,f} & \cdots & \theta^h_{d,F}\\
\vdots & \ddots & \vdots & \ddots & \vdots\\ 
\theta^h_{D,1} & \cdots & \theta^h_{D,f} & \cdots & \theta^h_{D,F}
\end{bmatrix}
\end{aligned}
$$

其中 $F$ 是隶属度函数的参数个数（**个人理解**）。如果隶属度函数采用 **高斯** 函数，那么参数为均值和方差（参数的个数为 2 ）。为了进行反向传播，必须要计算梯度，因此隶属度函数必须是连续的。（类似关于激活函数是否要求处处可导的问题，涉及次梯度，不做展开）

$\boldsymbol p^h$ 表示结论部分的系数矩阵，那么

$$
\begin{aligned}
\boldsymbol p^h = \begin{bmatrix}
p^h_{1,0} & \cdots & p^h_{1,f} & \cdots & p^h_{1,D}\\
\vdots & \ddots & \vdots & \ddots & \vdots\\ 
p^h_{r,0} & \cdots & p^h_{r,f} & \cdots & p^h_{r,D}\\
\vdots & \ddots & \vdots & \ddots & \vdots\\ 
p^h_{R,0} & \cdots & p^h_{R,f} & \cdots & p^h_{R,D}
\end{bmatrix}
\end{aligned}
$$

其中 $R$ 为规则个数。

对于输出层的 TSFN，其参数与隐层的 TSFN 类似，只不过将上标换为 $O$，即 $\boldsymbol \theta^o, \boldsymbol p^o$。

给定输入，隶属度函数的输出表示为

$$
\begin{aligned}
\boldsymbol \mu^h = \begin{bmatrix}
\mu^h_{1,1} & \cdots & \mu^h_{1,f} & \cdots & \mu^h_{1,D}\\
\vdots & \ddots & \vdots & \ddots & \vdots\\ 
\mu^h_{r,1} & \cdots & \mu^h_{r,d} & \cdots & \mu^h_{r,D}\\
\vdots & \ddots & \vdots & \ddots & \vdots\\ 
\mu^h_{R,1} & \cdots & \mu^h_{R,d} & \cdots & \mu^h_{R,D}
\end{bmatrix}
\end{aligned}
$$

其中 $\mu^h_{r,d}=\mu_{G^h_{r,d}(x_d)}$ 是第 $h$ 个TS模糊神经元中第 $r$ 个规则下第 $d$ 个输入的隶属度。

第 $r$ 个规则的权重计算如下（原文 t-norm ？）

<!-- $$
\begin{aligned}
\omega_r^h &= \land_{d=1}^D\mu_{r,d}^h\\
\boldsymbol \omega^h &= [\omega_1^h \cdots\omega_r^h \cdots\omega_R^h]^T
\end{aligned}
$$ -->

$$
\omega_r^h = \land_{d=1}^D\mu_{r,d}^h
$$

第 $r$ 个规则的输出（原文用 $v^h_r$）

<!-- $$
\begin{aligned}
y_r^h &= p^h_{r,0}+p^h_{r,1}x_1 + \cdots + p^h_{r,d}x_d+\cdots+p^h_{r,D}x_D\\
\boldsymbol y^h &= \boldsymbol p^h\times
\begin{bmatrix}
1\\
\boldsymbol x
\end{bmatrix}
\end{aligned}
$$ -->

$$
y_r^h = p^h_{r,0}+p^h_{r,1}x_1 + \cdots + p^h_{r,d}x_d+\cdots+p^h_{r,D}x_D
$$

最终 $S_h$ 的输出为

$$
a^h = \frac{\sum_{r=1}^R\omega^h_ry_r^h}{\sum_{r=1}^R\omega^h_r}
$$

$a^h$ 作为输出层的 STFN 的（$H$ 维）输入。

在输出层，经过上述类似的步骤，可以得到一个输出 $y^o$（原文用 $y^O$），作为整个 TSDFN 的输出，如下

$$
\begin{aligned}
\mu^o_{r,h} &=\mu_{G^o_{r,h}(a^h)}\\
\omega_r^o &= \land_{h=1}^H\mu_{r,h}^o\\
y_r^o &= p^o_{r,0}+p^o_{r,1}a^1 + \cdots + p^o_{r,h}a^h+\cdots+p^o_{r,H}a^H\\
y^o &= \frac{\sum_{r=1}^R\omega^o_ry_r^o}{\sum_{r=1}^R\omega^o_r}
\end{aligned}
$$

误差 $e = y^o-y_d$ 可用于 MSE 损失函数（原文 $\frac{1}{2n}$ 可能 **有误**）

$$
J = \frac{1}{2}\sum_{n=1}^N(e^{(n)})^2
$$

其中 $N$ 是数据样本（输入输出对）总个数，$e^n$ 是第 $n$ 个样本对应的误差。

### 3.2.2. 反向传播

首先求 loss 对**输出层参数**的梯度。

loss 对输出的梯度

$$
\frac{\partial J}{\partial y^o} = \sum_{n=1}^Ne^{(n)}
$$

loss 对输出系数 $p^o_{r,h}$ 的梯度

$$
\begin{aligned}
\frac{\partial J}{\partial p_{r,h}^o} &=\frac{\partial J}{\partial y^o}\frac{\partial y^o}{\partial y^o_r}\frac{\partial y^o_r}{\partial p^o_{r,h}} \\
&=\sum_{n=1}^Ne^{(n)}\cdot \frac{\omega^o_r}{\sum_{r=1}^R\omega^o_r}\cdot a^h
\end{aligned}
$$

其中 $a^0=1$ 。

loss 对输出层隶属度函数参数 $\theta^o_{h,f}$ 的梯度

$$
\frac{\partial J}{\partial \theta^o_{h,f}} =
\frac{\partial J}{\partial y^o}
\sum_{r=1}^R
\frac{\partial y^o}{\partial \omega^o_r}
\frac{\partial \omega^o_r}{\partial \mu^o_r}
\frac{\partial \mu^o_r}{\partial \theta^o_f}
$$

但是需要注意一点，不是每个隶属度函数都参与每条规则的计算（也即不是每个输入都参与规则计算）。假设有 $Q\leq R$ 个规则中包含待求解的隶属度函数的参数，则上式变为

$$
\frac{\partial J}{\partial \theta^o_{h,f}} =
\frac{\partial J}{\partial y^o}
\sum_{q=1}^Q(
\frac{\partial y^o}{\partial \omega^o_q}
\frac{\partial \omega^o_q}{\partial \mu^o_{q,h}}
\frac{\partial \mu^o_{q,h}}{\partial \theta^o_{h,f}}
)
$$

> $Q$ denotes the total number of rules in which the corresponding MF appears in premise part.
---

下面求 loss 对**隐层参数**的梯度。

首先求 loss 对隐层输出的梯度。注意到从 $y^o$ 到 $a^h$ 实际上是有两个部分的，因此下式包含两项

$$
\frac{\partial J}{\partial a^h} = 
\frac{\partial J}{\partial y^o}
\sum_{r=1}^R(
\frac{\partial y^o}{\partial y^o_h}
\frac{\partial y^o_h}{\partial a^h}+
\frac{\partial y^o}{\partial \omega^o_r}
\frac{\partial \omega^o_r}{\partial a^h}
)
$$

然后求 loss 对隐层系数 $p^h_{r,d}$ 和隐层隶属度函数参数 $\theta^h_{r,f}$ 的梯度

$$
\frac{\partial J}{\partial p_{r,d}^h} =
\frac{\partial J}{\partial a^h} \cdot
\frac{\partial a^h}{\partial y^h_r}
\frac{\partial y^h_r}{\partial p^h_{r,d}}
$$

$$
\frac{\partial J}{\partial \theta^h_{d,f}} =
\frac{\partial J}{\partial a^h}
\sum_{q=1}^Q(
\frac{\partial a^h}{\partial \omega^h_q}
\frac{\partial \omega^h_q}{\partial \mu^h_{q,d}}
\frac{\partial \mu^h_{q,d}}{\partial \theta^h_{d,f}}
)
$$

实际上，隐层和输出层的 $Q$ 应该用不同的符号表示，为了简略此处不加区分（**个人理解**）。

原文到此处就不再推导，这里也不进行展开了。

计算出全部梯度后，采用梯度下降更新参数。

## 实验

### 准备工作

采用高斯隶属度函数，简便起见，隐层的每个模糊神经元的每条规则中，均采用相同个数的隶属度函数。

为了从前提参数值评估规则权重，采用 **代数乘积**（而不是取小） 作为 t-norm 运算符，因为它易于求微分。

设计三种工况

- 不加任何处理的原始训练数据集；
- 增加小幅度的不精确的训练数据集。实际上，在进行测量时，总是存在测量值正确性的公差。实际值在测量值的指定公差范围内。为了在数据集中添加不精确度，数据集值的某一随机部分在某些带前缀（prefixed？）的公差带之间变化；
- 更进一步，添加模糊性使得数据集含糊不清。通常而言，模糊性是指对值的不清楚的理解或不正确和错误的度量。在模糊情况下，多个观测者（或传感器）对某个单个值没有达成共识。如果认为某数值属于模糊集，则更改其参数会导致模糊性（vagueness）增加到数据集中，因为更改模糊集的参数会由于模糊性而引入不确定性（uncertainty），并且模糊集的性质也会变得模糊（vague）。本文考虑使用高斯模糊集将模糊性添加到数据中。
  >  If data values are considered to belong to fuzzy set then varying its parameters leads to add vagueness into dataset since varying the parameter of fuzzy set introduces uncertainty due to vagueness and the nature of fuzzy set becomes vague.

后续每个实验中，针对上述每个情况，将数据集划分为 70% 的训练集，15% 的验证集和 15% 的测试集。

TSDFN 网络架构的确定包含下面几步：

- TSDFN 的网络架构：不同的隐层模糊神经元个数；
- TSDFN 的网络结构：不同的隶属度函数个数；
- 对每个网络结构在训练集上训练，在验证集上测试；
- 根据最好的测试性能确定隶属度函数个数（即确定网络结构）；
- 将已经确定隶属度函数个数的不同架构的网络在测试集上测试。（就能最终确定采用啥架构的 TSDFN 了）

在上述每个工况下，设计一个 3 层普通神经网络与 TSDFN 进行对比。神经网络激活函数采用 sigmoid 函数，隐层个数与 TSDFN 一致，训练方式采用现有的复杂方式（啥？）。

> This ANN is trained with existing sophisticated approaches.

采用 MSE 衡量性能。最终如图所示



# 4. TS 模糊控制

> T. Taniguchi; K. Tanaka; H. Ohtake; H.O. Wang. **Model construction, rule reduction, and robust compensation for generalized form of Takagi-Sugeno fuzzy systems**. IEEE Transactions on Fuzzy Systems ( Volume: 9, Issue: 4, Aug 2001).

在线性矩阵不等式（linear matrix inequality, LMI）设计框架下，基于 TS 模糊模型的非线性控制得以广泛应用。一般分为三个阶段：

- 第一阶段：对非线性被控对象的模糊建模
  - 利用输入输出数据进行模糊模型辨识
  - 或 基于分区非线性思想的模糊系统构建（模糊 IF-THEN 规则）
- 第二阶段：模糊控制规则推导，它反映了模糊模型的规则结构，它通过所谓的并行分布式补偿（PDC）实现
- 第三阶段：模糊控制器设计，即确定反馈增益。

> This paper presents a systematic procedure of fuzzy control system design that consists of fuzzy model construction, rule reduction, and robust compensation for nonlinear systems. 
 
本文提出了一种模糊控制系统设计的系统程序，该程序由模糊模型构建，规则约简和非线性系统的鲁棒补偿组成。


# 5. Fuzzy Control
> Robust ${L_1}$ Observer-Based Non-PDC Controller Design for Persistent Bounded Disturbed TS Fuzzy Systems


# 6. Trajectory Prediction
> Multi-agent Trajectory Prediction with Fuzzy Query Attention. NIPS 2020.

# 7. 参考文献

无。
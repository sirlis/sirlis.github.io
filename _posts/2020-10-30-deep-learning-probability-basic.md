---
title: 深度学习基础（概率与统计）
date: 2020-10-30 10:46:19 +0800
categories: [Academic, Knowledge]
tags: [probability]
math: true
---

本文主要介绍概率与统计的相关知识，包括概率的基本概念，似然函数，全概率，条件概率，贝叶斯公式，信息熵等概念的介绍。

<!--more-->

---
- [1. 基本概念](#1-基本概念)
  - [1.1. 概率定义](#11-概率定义)
  - [1.2. 随机变量](#12-随机变量)
  - [1.3. 概率分布与概率密度](#13-概率分布与概率密度)
  - [1.4. 概率和统计](#14-概率和统计)
  - [1.5. 概率函数与似然函数](#15-概率函数与似然函数)
  - [1.6. 极大似然估计](#16-极大似然估计)
- [2. 概率](#2-概率)
  - [2.1. 条件概率公式](#21-条件概率公式)
  - [2.2. 全概率公式](#22-全概率公式)
  - [2.3. 概率的两大学派](#23-概率的两大学派)
  - [2.4. 贝叶斯公式](#24-贝叶斯公式)
- [3. 熵](#3-熵)
  - [3.1. 自信息](#31-自信息)
  - [3.2. 信息熵](#32-信息熵)
  - [3.3. 相对熵（KL散度）](#33-相对熵kl散度)
  - [3.4. 交叉熵](#34-交叉熵)
  - [3.5. softmax 函数](#35-softmax-函数)
- [4. 参考文献](#4-参考文献)

# 1. 基本概念

## 1.1. 概率定义

**条件概率**：$P(A\vert B)$ 在某条件下事件发生的概率。

**先验概率**：指根据以往经验和分析得到的概率，如全概率公式，它往往作为"由因求果"问题中的"因"出现的概率。

**后验概率**：已知原分布，在实际发生某事件时,是原先某情况的可能性。后验概率是信息理论的基本概念之一。后验概率是指在得到“结果”的信息后重新修正的概率，是“执果寻因”问题中的"果"。先验概率与后验概率有不可分割的联系，后验概率的计算要以先验概率为基础。

事情还没有发生，要求这件事情发生的可能性的大小，是先验概率。事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小，是后验概率。

**后验概率是一种条件概率**。一种解释认为，条件概率是个数学名称，后验概率是建模的时候赋予了一定的意义。一般的条件概率，条件和事件可以是任意的；对于后验概率，它限定了事件为隐变量取值，而条件为观测结果。

**联合概率**：$P(AB)$，表示两个事件共同发生的概率。

**边缘概率**：是某个事件发生的概率，而与其它事件无关。在联合概率中，把最终结果中不需要的那些事件合并成其事件的全概率而消失（对离散随机变量用求和得全概率，对连续随机变量用积分得全概率）。这称为边缘化（marginalization）。$A$ 的边缘概率表示为 $P(A)$，$B$ 的边缘概率表示为 $P(B)$。

需要注意的是，在这些定义中 $A$ 与 $B$ 之间不一定有因果或者时间顺序关系。$A$ 可能会先于 $B$ 发生，也可能相反，也可能二者同时发生。$A$ 可能会导致 $B$ 的发生，也可能相反，也可能二者之间根本就没有因果关系。

## 1.2. 随机变量

产品经理马忠信. [应该如何理解概率分布函数和概率密度函数？](https://www.jianshu.com/p/b570b1ba92bb)

> 微积分是研究**变量**的数学，概率论与数理统计是研究**随机变量**的数学。

> 研究一个随机变量，不只是要看它能取哪些值，更重要的是它取各种值的**概率**如何。

随机变量（random variable）表示随机试验各种结果的实值单值函数。随机事件不论与数量是否直接有关，都可以数量化，即都能用数量化的方式表达。

随机事件数量化的好处是可以用数学分析的方法来研究随机现象。例如某一时间内公共汽车站等车乘客人数，电话交换台在一定时间内收到的呼叫次数，灯泡的寿命等等，都是随机变量的实例。

按照随机变量可能取得的值，可以把它们分为两种基本类型：

- **离散型随机变量**：随机变量的值可以逐个列举。
  - 离散型随机变量通常依据概率质量函数分类，主要分为：伯努利随机变量、二项随机变量、几何随机变量和泊松随机变量。
  - 离散型随机变量的期望为 $\mathbb E[X] = \sum_{x:p(x)>0}xp(x)$。
- **连续型随机变量**：随机变量的取值无法逐个列举。
  - 有几个重要的连续随机变量常常出现在概率论中，如：均匀随机变量、指数随机变量、伽马随机变量和正态随机变量。
  - 连续型随机变量的期望为 $\mathbb E[X] = \int_{-\infty}^{+\infty}xf(x)dx$（一阶矩）。「见后文推导」

## 1.3. 概率分布与概率密度

> 对于离散型随机变量，其概率可以用概率分布列表来描述。如两个离散型随机变量 $x$ 的概率分布列表可以表述为
> $$
> \begin{bmatrix}
> x\\ 
> P(x)
> \end{bmatrix} = \begin{bmatrix}
> x_1 & x_2\\ 
> 0.99 & 0.01
> \end{bmatrix}
> $$
> 解读为：$x$ 取值为 $x_1$ 的概率 $P(x=x_1) = 0.99$，取值为 $x_2$ 的概率 $P(x=x_2) = 0.01$。

**概率分布函数**（累积概率函数）是描述随机变量取值分布规律的数学表示。概率分布函数是随机变量特性的表征，它决定了随机变量取值的分布规律，只要已知了概率分布函数，就可以算出随机变量落于某处的概率。

**对于离散型随机变量，概率分布函数定义为**

$$
F(x) = P(x\leq x_i)=\sum_{x\leq x_i} P(x_i)
$$

对于连续型随机变量，设变量 $x$ 的取值区间为 $(a,b)$，其概率分布函数为

$$
F(x) = P(a<x<b) = F(b) - F(a)
$$

引入 “**概率密度函数** $f(x)$” 的概念，定义为概率分布函数的导数，是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。即

$$
f(x) = F'(x) = \mathop{lim}\limits_{\Delta x\rightarrow 0}\frac{F(x+\Delta x) - F(x)}{\Delta x}
$$

概率密度函数的性质有

- $f(x) \geq 0$
- $\int_{-\infty}^{+\infty} f(x)dx=1$

随机变量的取值落在某个区域之内的概率则为概率密度函数在这个区域上的积分。当概率密度函数存在的时候，概率分布函数是概率密度函数的积分。

**对于连续型随机变量，其概率分布函数为**

$$
F(x) = \int_{-\infty}^{x_i} f(x)dx
$$

那么前面连续型随机变量 $x\in(a,b)$ 的概率分布函数为

$$
F(x) = P(a<x<b) = F(b) - F(a) =\int_{a}^{b} f(x)dx
$$

连续型随机变量 $X$ 的**期望** $\mathbb E(X)$ 为

$$
\mathbb E(X) = \int_{-\infty}^{+\infty}xf(x)dx
$$

可以通过将概率密度函数划分为 $n$ 个小区间，对每个小区间求概率分布表，当小区间个数 $n \rightarrow +\infty$ 时，$max(\Delta x_i)\rightarrow 0$，借助对离散型随机变量的期望求极限，得到上述定义式。即

$$
\mathbb E(X) = \mathop{lim}\limits_{n\rightarrow+\infty}\sum_{i=1}^n x_if(x_i)\Delta x_i =  \int_{-\infty}^{+\infty}xf(x)dx
$$

## 1.4. 概率和统计

概率（probabilty）和统计（statistics）看似两个相近的概念，其实研究的问题刚好相反。

概率研究的问题是，已知一个模型和参数，怎么去预测这个模型产生的结果的特性（例如均值，方差，协方差等等）。 举个例子，我想研究怎么养猪（模型是猪），我选好了想养的品种、喂养方式、猪棚的设计等等（选择参数），我想知道我养出来的猪大概能有多肥，肉质怎么样（预测结果）。

统计研究的问题则相反。统计是，有一堆数据，要利用这堆数据去预测模型和参数。仍以猪为例。现在我买到了一堆肉，通过观察和判断，我确定这是猪肉（这就确定了模型。在实际研究中，也是通过观察数据推测模型是／像高斯分布的、指数分布的、拉普拉斯分布的等等），然后，可以进一步研究，判定这猪的品种、这是圈养猪还是跑山猪还是网易猪，等等（推测模型参数）。

一句话总结：概率是已知模型和参数，推数据。统计是已知数据，推模型和参数。

统计领域有两个基本概念：**极大似然估计** 和 **极大后验概率估计**。它们都是用来推测参数的方法。为什么会存在着两种不同方法呢？ 这需要理解概率的两个学派。（很奇怪的一点在于，两个估计是属于统计领域，但是确需要理解概率的两个学派）

## 1.5. 概率函数与似然函数

似然（likelihood）这个词其实和概率（probability）是差不多的意思，Colins字典这么解释：The likelihood of something happening is how likely it is to happen. 你把likelihood换成probability，这解释也读得通。但是在统计里面，似然函数和概率函数却是两个不同的概念（其实也很相近就是了）。

对于 $P(x\vert \theta)$，输入有2个：$x$ 表示某一具体的数据，$\theta$ 表示模型参数。

- 如果 $\theta$ 是已知的，$x$ 是变量，这个函数叫做**概率函数**，它描述对于不同样本点 $x$，其出现的概率是多少。

- 如果 $x$ 是已知的，$\theta$ 是变量，这个函数叫做**似然函数**，它描述对于不同的模型参数，出现 $x$ 这个样本点的概率是多少。

大概直观的理解一下，假设某组数据 $x$ 服从正态分布，其概率密度函数为

$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

> 说到这里，插一个啼笑皆非的事情，在推免研究生面试时，一帮教授大佬们突然恶趣味想出这么一个面试题 “高斯分布和正态分布是什么关系？”，答案是：“它们是一个东西”。结果在面试期间，大佬们真的对好几个学生都问了这个问题。尴尬的是，只有一个学生犹犹豫豫的回复说：“它们好象是一个东西把...”

- 如果我们已知分布的参数 $\theta = [\mu,\sigma]$ 的具体取值，那么 $P(x\vert \theta)$ 就是取到数据 $x$ 的概率。
- 如果我们不知道分布的参数，而是通过一次采样得到数据 $x$，那么 $P(x\vert \theta)$ 描述的就是对于不同的模型参数，出现数据 $x$ 这个样本点的概率。

注意到，上面两种不同的解读与概率的两个学派紧密相联。

## 1.6. 极大似然估计

> 忆臻. [一文搞懂极大似然估计](https://zhuanlan.zhihu.com/p/26614750)
> 拓季. [交叉熵与最大似然估计](https://www.jianshu.com/p/191c029ad369)
> 贺勇，明杰秀编著．概率论与数理统计．武汉：武汉大学出版社，2012.08：216-217

极大似然估计（Maximum likelihood estimation, MLE），是建立在极大似然原理的基础上的一个统计方法。通俗理解来说，就是利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值。

在现实世界的研究中，我们可以首先假定某些被研究对象的概率分布服从某种分布形态，例如二项分布、正态分布、指数分布，但对于相应分布中的具体参数的取值却无从知晓，此时我们可以借助从总体中抽样的方式，对总体的概率分布的参数进行一个点对点的估计。

> 一种直观的理解：设甲箱中有 99 个白球，1 个黑球；乙箱中有 1 个白球．99 个黑球。现随机取出一箱，再从抽取的一箱中随机取出一球。假设取出的结果是黑球，由于这一黑球从乙箱抽取的概率比从甲箱抽取的概率大得多，这时我们自然更多地相信这个黑球是取自乙箱的。

极大似然估计中采样需满足一个重要的假设，就是抽样样本集 $X$ 当中的样本 $x_1,x_2,...,x_n$ 彼此**独立同分布**。一般说来，事件 $x_i$ 发生的概率与未知的模型参数 $\theta$ 有关，参数 $\theta$ 的取值不同，则事件 $x_i$ 发生的概率 $P(x_i\vert \theta)$ 也不同。当我们在一次试验中事件 $x_i$ 发生了，则认为此时的 $\theta$ 值应是的一切可能取值中使 $P(x_i\vert \theta)$ 达到最大的那一个，极大似然估计法就是要选取这样的值作为参数 $\theta$ 的估计值，使所选取的样本在被选的总体中出现的可能性为最大。

对于样本集 $X$，由于样本之间彼此独立且服从参数为 $\theta$ 的分布，那么取得任意一个样本 $x_i$ 的概率可以表示为 $P(x_i \vert \theta)$，因而取得当前这个样本集的概率为：

$$
P(X) = P(x_1\vert \theta)P(x_2\vert \theta)...P(x_n\vert \theta)
$$

由于在抽样后会得到总体的一个观测，因此相应的概率在具体的样本集中都是已知的，上面这个函数则可以看作在不同的参数 $\theta$ 的取值下，取得当前这个样本集的概率，也即可能性（likelihood），因此将其称为参数 $\theta$ 相对于样本集 $X$ 的**似然函数（likelihood function）**，记为 $L(\theta)$，即有

$$
\begin{aligned}
L(\theta) & = L(x_1,x_2,...,x_n\vert \theta)\\
&= P(x_1\vert \theta)P(x_2\vert \theta)...P(x_n\vert \theta)\\
&= \prod P(x_i\vert \theta), i=1,2,...,n
\end{aligned}
$$

**对于抽样得到样本集 $X$ 这个既成事实，我们认为这就是最该发生的结果，也即所有可能的结果中概率最大的那一个，这就是最大似然估计的命名由来**。此时，我们需要寻找的就是那个使似然函数取得最大值的 $\theta$ 值

$$
{\rm argmax}\ L(\theta) = {\rm max}_\theta L(\theta)
$$

${\rm argmax} f(x)$ 是使得函数 $f(x)$ 取得其最大值的所有自变量 $x$ 的集合。

总结起来，利用最大似然函数求解总体参数的估计值的一般步骤为：

- 获取似然函数
- 对似然函数取自然对数
- 将对数似然函数求（偏）导数，令其为 0，得到似然方程
- 求解似然方程，得到的一个或多个数值，即为总体参数的最大似然估计值

---

由于在计算当中，多个概率的乘法最终会得到一个非常小的值，从而可能造成下溢（underflow），因此一般会对似然函数取一个对数，将连续乘法转化为加法

$$
{\rm argmax}\ log L(\theta) = {\rm argmax}\  log \prod P(x_i\vert \theta) = {\rm argmax}\  \sum_{i=1}^n logP(x_i\vert \theta), i=1,2,...,n
$$

再进一步，加一个负号，问题转化为

$$
{\rm argmin}\  [-log L(\theta)] = {\rm argmin}\  \sum_{i=1}^n [-logP(x_i\vert \theta)], i=1,2,...,n
$$

因为当重新缩放代价函数时 argmin 不会改变，我们可以除以 $n$ 得到和训练数据经验分布 $P$ 相关的期望作为准则

$$
{\rm argmin}\ \mathbb E_P [-logP(X\vert \theta)]
$$

令 $Q(X) = P(X\vert \theta)$ 上式转化为

$$
{\rm argmin}\ \mathbb E_P [-log Q(X)]
$$

后面在介绍 KL 散度和交叉熵时会发现，**任何一个负对数似然组成的损失都是定义在训练集上的经验分布和定义在模型上的真实分布之间的交叉熵。**

---

**举一个例子**：假定被研究对象的总体服从二项分布，如果知道了基础的 (0，1) 分布中结果指定为 1 时的概率 $p$，也就知道了这个分布形态的全部。

> 假如有一个罐子，里面有黑白两种颜色的球，数目多少不知，两种颜色的比例也不知。我们想知道罐中白球和黑球的比例，但我们不能把罐中的球全部拿出来数。现在我们可以每次任意从已经摇匀的罐中拿一个球出来，记录球的颜色，然后把拿出来的球再放回罐中。这个过程可以重复，我们可以用记录的球的颜色来估计罐中黑白球的比例。假如在前面的一百次重复记录中，有七十次是白球，请问罐中白球所占的比例最有可能是多少？
> 
> 很多人马上就有答案了：**70%**。而其后的理论支撑是什么呢？
> 
> 我们假设罐中白球的比例是 $p$，那么黑球的比例就是 $1-p$，这里 $p$ 就是模型参数。因为每抽一个球出来，在记录颜色之后，我们把抽出的球放回了罐中并摇匀，所以每次抽出来的球的颜色服从同一独立分布。
> 
> 这里我们把一次抽出来球的颜色称为一次抽样。题目中在一百次抽样中，七十次是白球的,三十次为黑球事件的概率是：
> $$
> P(样本结果|Model)
> $$
> 如果第一次抽象的结果记为 $x_1$，第二次抽样的结果记为 $x_2$...，那么样本结果为 $(x_1,x_2,...,x_{100})$。这样，我们可以得到如下表达式：
> $$
> \begin{aligned}
> P(样本结果|Model)&= P(x_1,x_2,…,x_{100}|Model)\\
> &= P(x_1|M)P(x_2|M)\cdots P(x_{100}|M)\\
> &= p^{70}(1-p)^{30}
> \end{aligned}
> $$
> 上面就是观察样本结果出现的概率函数了。那么，如何求出模型参数 $p$ 呢？不同的 $p$ 会直接导致上面概率函数的不同。比如:
> - $p=0.5$（黑50%白50%）时 $p^{70}(1-p)^{30}=7.8\times 10^{-31}$
> - $p=0.7$ （黑70%白30%）时 $p^{70}(1-p)^{30}=2.95\times 10^{-27}$
> 
> 那么问题来了，既然有无数种分布可以选择，极大似然估计应该按照什么原则去选取这个分布呢？答案是，**让这个样本结果出现的可能性最大**，既然事情已经发生了，为什么不让这个出现的结果的可能性最大呢？这也就是最大似然估计的核心。转化为数学问题，就是使得 $P(样本结果|Model)=p^{70}(1-p)^{30}$ 的值最大。
> 
> 如何求极值呢，那么我们就可以将概率函数看成是 $p$ 的方程，对其**求导**，导数为 0 的点就是极值点！
> $$
> \begin{aligned}
> f'(p) &= {\rm d}(p^{70}(1-p)^{30})/{\rm d}p\\
> &= 70\cdot p^{69}(1-p)^{30}-p^{70}\cdot 30\cdot(1-p)^{29}\\
> &= 70\cdot p^{69}\cdot(1-p)^{29}[1-p-3/7\cdot p] = 0\\
> \end{aligned}
> $$
> 即
> $$
> \begin{aligned}
> 1 - 10/7 \cdot p = 0
> \end{aligned}
> $$
> 可求出 $p=0.7$，与我们一开始认为的 70% 是一致的。

**再举另外一个例子**

> 假设我们要统计全国人民的年均收入，首先假设这个收入服从服从正态分布，但是该分布的均值与方差未知。我们没有人力与物力去统计全国每个人的收入。我们国家有10几亿人口呢？那么岂不是没有办法了？
> 
> 不不不，有了极大似然估计之后，我们可以采用嘛！我们比如选取一个城市，或者一个乡镇的人口收入，作为我们的观察样本结果。然后通过最大似然估计来获取上述假设中的正态分布的参数。
> 
> 有了参数的结果后，我们就可以知道该正态分布的期望和方差了。也就是我们通过了一个小样本的采样，反过来知道了全国人民年收入的一系列重要的数学指标量！
> 
> 那么我们就知道了极大似然估计的核心关键就是对于一些情况，样本太多，无法得出分布的参数值，可以采样小样本后，利用极大似然估计获取假设中分布的参数值。

# 2. 概率

## 2.1. 条件概率公式

条件概率是指事件 $A$ 在事件 $B$ 发生的条件下发生的概率。

若只有两个事件 $A$，$B$，那么条件概率表示为：$P(A\vert B)$，读作 “$A$ 在 $B$ 发生的条件下发生的概率”。

$$
P(A\vert B) = \frac{P(AB)}{P(B)}
$$

其中，$P(AB)$ 是两个事件的联合概率，$P(B)$ 是事件 $B$ 的边缘概率。

**一种不准确的理解方式**，$P(AB)$ 是 $A,B$ 同时发生的概率，$P(B)$ 是 $B$ 发生的概率。我们要求已知 $B$ 发生后 $A$ 发生的概率，相当于 B 的发生与否不再是一个不确定的概率，而是确定的条件。由于不同事件同时发生的概率是用乘积的形式来表示，反之某个事件从概率变为已经发生就用除法来表示，那么就应该用 $P(AB)$ 除以 $P(B)$ 来得到这么个条件概率。

**另一种理解方式**，将等式做一个变换

$$
P(B)P(A\vert B) = P(AB)
$$

也就是说，$A,B$ 同时发生的概率，等于 $B$ 发生的概率，乘以 $B$ 发生（作为条件）后 $A$ 发生的概率。

这给我们一个启示，即交换 $A,B$ 的顺序，等式依然成立

$$
P(B)P(A\vert B) = P(AB) = P(BA) = P(A)P(B\vert A)
$$

## 2.2. 全概率公式

设 $B_1,...,A_n$ 是样本空间 $S$ 的一个完备事件组，即

- $B_1,...,B_n$ 两两不相容：$B_i \cap B_j = \varnothing\quad (i\neq j)$
- $B_i \cup...\cup B_n = S$

每一次试验中，完备事件组中有且仅有一个事件发生。完备事件组构成样本空间的一个划分。

**全概率公式**。定理：设实验 $E$ 的样本空间为 $S$，$B_1, B_2,...,B_n$ 为 $S$ 的一个划分（完备事件组），且 $P(B_i)>0\quad i=1,2,...n$，$A$ 为 $E$ 的一个事件，则

$$
\begin{aligned}
P(A) &= P(B_1)P(A\vert B_1)+P(B_2)P(A\vert B_2)+...+P(B_n)P(A\vert B_n)\\
&= \sum_{i=1}^n P(B_i)P(A\vert B_i)
\end{aligned}
$$

其推导过程如下，如图：

![full probability](../assets/img/postsimg/20201030/1.jpg)

$$
\begin{aligned}
A &= AS = A(B_1\cup B_2\cup ... \cup B_n)\\
&= AB_1\cup AB_2\cup ... \cup AB_n \quad (AB_i两两互斥)\\
P(A) &= P(AB_1\cup AB_2\cup ... \cup AB_n)\\
&=P(AB_1) + P(AB_2) + ... + P(AB_n)
\end{aligned}
$$

根据条件概率公式

$$
P(AB_i) = P(A)P(B_i\vert A) = P(B_i)P(A\vert B_i)
$$

带入有

$$
P(A) = \sum_{i=1}^n P(B_i)P(A\vert B_i)
$$

即为全概率公式。

**全概率公式的意义1**：将复杂的事件 $A$ 划分为比较简单的事件 $AB_1,...,AB_n$，再结合加法和乘法计算 $A$ 的（边缘）概率。

**全概率公式的意义2**：事件 $A$ 的发生可能有各种原因 $B_i\quad (i=1,2,...,n)$，如果 $A$ 是由 $B_i$ 引起，则此时 $A$ 发生的（条件）概率为

$$
P(AB_i) = P(B_i)P(A\vert B_i)
$$

若每个原因都可能导致 $A$ 的发生，那么 $A$ 发生的概率是全部原因引起其发生的概率的综合，即为全概率公式。

因此可以把全概率公式看成是 “**由原因推结果**”。每一个原因对结果的发生由一定的作用，结果发生的可能性与各种原因的作用大小有关，全概率公式表达了它们之间的关系。

## 2.3. 概率的两大学派

对于概率看法不同的两大派别频率学派与贝叶斯派。他们看待世界的视角不同，导致他们对于产生数据的模型参数的理解也不同。

- **频率学派**

  - 他们认为世界是确定的。他们直接为事件本身建模，也就是说事件在多次重复实验中趋于一个稳定的值 $p$，那么这个值就是该事件的概率。
  - 他们认为模型参数是个定值，希望通过类似解方程组的方式从数据中求得该未知数。这就是频率学派使用的参数估计方法：**极大似然估计**（MLE），这种方法往往在大数据量的情况下可以很好的还原模型的真实情况。
  - 频率派把需要推断的模型参数 $\theta$ 看做是固定的未知常数，即参数 $\theta$ 虽然是未知的，但最起码是确定的一个值，同时，样本 $X$ 是随机的，所以频率派重点研究样本空间，大部分的概率计算都是针对样本 $X$ 的分布；

- **贝叶斯派**

  - 他们认为世界是不确定的，因获取的信息不同而异。假设对世界先有一个预先的估计，然后通过获取的信息来不断调整之前的预估计。 他们不试图对事件本身进行建模，而是从旁观者的角度来说。因此对于同一个事件，不同的人掌握的先验不同的话，那么他们所认为的事件状态也会不同。
  - 他们认为模型参数源自某种潜在分布，希望从数据中推知该分布。对于数据的观测方式不同或者假设不同，那么推知的该参数也会因此而存在差异。这就是贝叶斯派视角下用来估计参数的常用方法：**最大后验概率估计**（MAP），这种方法在先验假设比较靠谱的情况下效果显著，随着数据量的增加，先验假设对于模型参数的主导作用会逐渐削弱，相反真实的数据样例会大大占据有利地位。极端情况下，比如把先验假设去掉，或者假设先验满足均匀分布的话，那她和极大似然估计就如出一辙了。
  - 贝叶斯学派认为，模型参数 $\theta$ 是随机变量，而样本 $X$ 是固定的，由于样本是固定的，所以他们重点研究的是参数 $\theta$ 的分布。

贝叶斯学派的创始人，托马斯·贝叶斯（Thomas Bayes），发表了一篇名为《An essay towards solving a problem in the doctrine of chances》，翻译过来则是《机遇理论中一个问题的解》。这篇论文发表后，在当时并未产生多少影响，在20世纪后，这篇论文才逐渐被人们所重视。

贝叶斯派既然把 $\theta$ 看做是一个随机变量，所以要计算 $\theta$ 的分布，便得事先知道 $\theta$ 的无条件分布，即在有样本之前（或观察到 $X$ 之前）， $\theta$ 有着怎样的分布呢？

比如往台球桌上扔一个球，这个球落会落在何处？如果是不偏不倚的把球抛出去，那么此球落在台球桌上的任一位置都有着相同的机会，即球落在台球桌上某一位置的概率服从均匀分布。这种在实验之前定下的属于基本前提性质的分布称为先验分布，或的无条件分布。

至此，贝叶斯及贝叶斯派提出了一个思考问题的固定模式：

$$
先验分布 \pi(\theta) + 样本信息x \Rightarrow 后验分布 \pi(\theta \vert x)
$$

## 2.4. 贝叶斯公式

- **一种形式**

设 $B_1,...,B_n$ 是样本空间 $S$ 的一个完备事件组，则对任一事件 $A$，$P(A)>0$，有

$$
P(B_i\vert A) = \frac{P(B_i)P(A\vert B_i)}{P(A)}=\frac{P(B_i)P(A\vert B_i)}{\sum_{j=1}^n P(B_j)P(A\vert B_j)}
$$

贝叶斯公式的推导可以通过条件概率公式得到

$$
\begin{aligned}
P(A\vert B_i) &= \frac{P(AB_i)}{P(B_i)} \quad &(条件概率公式)\\
\Rightarrow  P(AB_i) &= P(B_i)P(A\vert B_i) \quad &(移项)\\
\Rightarrow  P(A)P(B_i\vert A) &= P(B_i)P(A\vert B_i) \quad &(条件概率公式)\\
\Rightarrow  P(B_i\vert A) &= \frac{P(B_i)P(A\vert B_i)}{P(A)} \quad &(移项)
\end{aligned}
$$

一种直观的理解：假设 $B_1,...,B_n$ 是发生时间 $A$ 的各种原因，那么已知事件 $A$ 已经发生了，问是原因 $B_i$ 导致的概率 $P(B_i \vert A)$ 是多少？

- **另一种形式**

使用另一套字母体系：
- $H$：hypothesis，假设，规律
- $E$：evidence，证据，现象，数据
 
那么贝叶斯的推理过程可以表达为：通过不断收集证据 $E$ 来完善假设 $H$ 。那么贝叶斯公式可以描述为

$$
P(H\vert E) = \frac{P(H)P(E\vert H)}{P(E)} = P(H)\frac{P(E\vert H)}{P(E)}
$$

因此贝叶斯公式实际上阐述的是如下事实

$$
新信息E出现后假设H的概率 = H 的概率\times 新信息带来的调整
$$

可以理解为，新观察到的样本信息将修正人们以前对事物的认知。

对于一个场景，可能有几种不同的规律来解释，根据对场景的一些现象的观测，如何知道各种可能的规律在背后发生作用的概率？这个问题就是如何求解 $P(规律H\vert现象E)$。

直接去计算有难度，但是如果我们知道在某个规律 $H$ 下，不同现象 $E$ 发生的概率 $P(现象E|规律H)$，和每个规律发生的概率 $P(规律H)$，和不同现象发生的概率 $P(现象E)$，就可以通过已知数据求解。贝叶斯就是告诉怎么用这些知道的知识去**计算现象后面的规律发生的概率**。

- **第三种形式**

马同学. [如何理解贝叶斯推断和beta分布？](https://www.matongxue.com/madocs/910)

假设实验数据 $X\vert p$ 服从二项分布（比如抛硬币）

$$
\underbrace{f(p\vert X=k)}_{后验分布}=\frac{\overbrace{P(X=k\vert p)}^{实验数据}\overbrace{f(p)}^{先验分布}}{\underbrace{P(X=k)}_{常数}}
$$

其中 $k$ 为正面的次数。分母与实验数据无关，可以视作常数。

# 3. 熵

《信息论：基础理论与应用》

## 3.1. 自信息

在信息传输的一般情况下，收信者获得的信息量等于信息传输前后不确定性的减少量

$$
获得的信息量 = 不确定性的减少量 = 收信前某事件发生的不确定性 - 收信后某事件发生的不确定性
$$

假设无噪声，信道传输不失真，因此收到消息后某事件发生的不确定性完全消除，即

$$
获得的信息量 = 不确定性的减少量 = 收信前某事件发生的不确定性 = 信源输出某消息中含有的信息量
$$

事件发生的不确定性与事件发生的概率有关。事件发生的概率越大，该事件发生后能够提供的信息就越少，不确定性就越小。对于发生概率等于 1 的事件（必然事件），提供不了任何信息，就不存在不确定性。因此，某事件发生所含有的信息量就应该是该事件发生先验概率的**函数**。

$$
I(a_i) = f(P(a_i))
$$

式中，$P(a_i)$ 是事件 $a_i$ 发生的先验概率，$I(a_i)$ 表示事件 $a_i$ 发生所含有的信息量，我们称之为 $a_i$ 的**自信息**。

下面分析自信息与先验概率之间函数关系 $f$ 的具体形式。

> 一种形象理解：事件发生概率越大，发生后能够提供的信息就越少（老发生没啥价值，获得不了什么信息），那么不确定性就越小，自信息就小。反之，如果事件发生概率越小，一旦发生了提供的信息量就很大，发生后消除的不确定性就很大，自信息量就大。因此，概率和信息量可能是负相关的。

根据客观事实和人们的习惯概念，不确定性函数 $f$ 应该满足如下的条件：

- **减函数**：$f$ 是 $P$ 的单调递减函数，
  - $P(a_i)=1$ 时 $f(P_i)=0$，
  - $P(a_i)=0$ 时 $f(P_i)=\infty$
- **可加性**：两个独立消息所产生的不确定性应等于各自不确定性之和，$f(P_1,P_2)=f(P_1)+f(P_2)$
- 由于事件 $P_1,P_2$ 是独立不相关的，因此 $P(1,2)=P_1P_2$。

同时满足这 3 个条件的函数 $f$ 是**负对数函数**，即

$$
f(P) = {\rm log}\frac{1}{P} = -{\rm log}P\quad P \in [0,1]
$$

函数图像如下图所示

![-log](../assets/img/postsimg/20201030/2.jpg)

**事件发生前，$f$ 表示事件发生的不确定性；事件发生后，$f$ 表示事件所含有（所提供）的信息量**。自信息采用的单位取决于对数所选的底。由于概率 $P$（作为自变量） 是小于 1 的正数，又根据实际情况自信息也必然是正数，所以对数的底应选取为大于 1 的任意数。

信息论中一般取 2 为底，则所得的信息量单位为 **比特**（bit，binary unit）。机器学习中一般取 $e$ 为底的自然对数，信息量单位为 **奈特**（nat, nature unit）。如果取 10为底，信息量单位为 **哈特**（hart, Hartley 的缩写，纪念他首先提出用对数来度量信息）。

取 2 为底时可以略去不写。如果 $P(a_i)=0.5$，则 $f(P)=1$ 比特。所以 1 比特信息量就是两个互不相容的等可能事件之一发生时提供的信息量。注意，这里的比特时抽象的信息量单位，与计算机术语中的比特含义不同，它代表二元数字（binary digits）。这两种概念之间的关系是，每个二元数字所能提供的最大平均信息量为 1 比特。

## 3.2. 信息熵

前面定义的自信息是某一信源发出某一消息所含有的信息量。如果发出的消息不同，所含有的信息量也就不同。因此，自信息是一个随机变量，不能用来作为整个信源的信息测度。

我们定义**自信息的数学期望为信源的平均自信息量**。信源一般包括多种信号，如果考虑这个信源所有可能发生情况的平均不确定性，假设信源消息有 $n$ 种取值：$x_1,...,x_i,...,x_n$，对应概率为：$P(x_1),...,P(x_i),...,P(x_n)$，且各种消息的出现彼此独立。这时，信源的平均不确定性应当为单个消息不确定性 $-{\rm log}P(x_i)$ 的数学期望（$\mathbb E$），可称为**信息熵**

$$
H(X) = \mathbb E[-{\rm log} P(x_i)] = -\sum_{i=1}^n P(x_i){\rm log} P(x_i)
$$

**信息熵：信源的平均不确定性，为单个消息不确定性的数学期望（统计平均值）**。1948年，C.E.Shannon（香农）提出了 “信息熵” 的概念，才解决了对信息的量化度量问题。信息熵这个词是香农从热力学中借用过来的。热力学中的热熵是表示分子状态混乱程度的物理量。

对于连续分布，信息熵的表示如下（连续型随机变量的数学期望）

$$
H(X) = \mathbb E[-{\rm log} P(x)] = -\int_x P(x){\rm log} P(x) {\rm d}x
$$

香农用**信息熵的概念来描述信源的平均不确定度，相当于每个消息所提供的平均信息量**。

例如，两个信源分别为

$$
\begin{bmatrix}
X\\ 
P(x)
\end{bmatrix} = \begin{bmatrix}
a_1 & a_2\\ 
0.99 & 0.01
\end{bmatrix},\quad \begin{bmatrix}
Y\\ 
P(y)
\end{bmatrix} = \begin{bmatrix}
b_1 & b_2\\ 
0.5 & 0.5
\end{bmatrix}
$$

两个信源的信息熵分别为

$$
\begin{aligned}
  H(X) &= -0.99{\rm log} 0.99-0.01{\rm log} 0.01 = 0.08 (bit/info)\\
  H(Y) &= -0.5{\rm log} 0.5-0.5{\rm log} 0.5 = 1 (bit/info)
\end{aligned}
$$

可见，信源 $Y$ 的平均不确定性大。我们观察信源 $Y$，它的两个输出消息是等可能性的，所以在没有输出前猜测输出哪个消息的不确定性要大。反之，对于信源 $X$，由于事件 $a_1$ 出现的概率远超事件 $a_2$，虽然具有不确定性，但是大致猜测事件 $a_1$ 会出现。

举个例子说明信息熵的作用。

> 赌马比赛，有4匹马 $\{ A, B, C, D\}$，获胜概率分别为 $\{ 1/2, 1/4, 1/8, 1/8 \}$，将哪一匹马获胜视为随机变量X属于 $\{ A, B, C, D \}$ 。
> 假定我们需要用尽可能少的二元问题来确定随机变量 X 的取值。
> 例如，问题1：A获胜了吗？　问题2：B获胜了吗？　问题3：C获胜了吗？
> 最后我们可以通过最多3个二元问题，来确定取值：
> - 如果X = A，那么需要问1次（问题1：是不是A？），概率为1/2 
> - 如果X = B，那么需要问2次（问题1：是不是A？问题2：是不是B？），概率为1/4 
> - 如果X = C，那么需要问3次（问题1，问题2，问题3），概率为1/8 
> - 如果X = D，那么需要问3次（问题1，问题2，问题3），概率为1/8 

在二进制计算机中，一个比特为 0 或 1，其实就代表了一个二元问题的回答。我们可以使用一个 2 比特的数字来完成上面的赌马结果：

```
00 - A获胜，  01 - B获胜，  10 - C获胜，  11 - D获胜
```

然而，我们可以利用非均匀分布这个特点，使用更短的编码来描述更可能的事件，使用更长的编码来描述不太可能的事件。我们希望这样做能够得到一个更短的平均编码长度。我们可以使用下面的编码串（哈夫曼编码）：

```
0 - A获胜,   10 - B获胜,   110 - C获胜,   111 - D获胜
```

此时，传输的编码的平均长度就是：

$$
1/2\times 1bit + 1/4\times 2bit + 1/8\times 3bit + 1/8\times 3bit = \frac{7}{4}bits = 1.75bits
$$

回到信息熵的定义，会发现通过之前的信息熵公式，神奇地得到了

$$
H(N) = \frac{1}{2}\cdot {\rm log}(2)+\frac{1}{4}\cdot {\rm log}(4)+\frac{1}{8}\cdot {\rm log}(8)+\frac{1}{8}\cdot {\rm log}(8) =\frac{1}{2}+\frac{2}{4}+\frac{3}{8}+\frac{3}{8} = \frac{7}{4} bits
$$

也就是说，在计算机中，我们给哪一匹马夺冠这个事件进行编码，所需要的平均码长为 1.75 个比特。所以，信息熵 $H(X)$ 可以看做**对信源 $X$ 中的样本进行编码所需要的编码长度的期望值**，同时也表明，信息熵是传输一个随机变量状态值所需的比特位下界（最短平均编码长度）。

## 3.3. 相对熵（KL散度）

相对熵，relative entropy = Kullback-Leibler divergence，用于衡量两个概率分布之间的差异。

设 $P(x)$ 和 $Q(x)$ 是离散随机变量 $X$ 中取值的两个概率分布。其中，$P(x)$ 为我们往往不能测得的真实分布，而 $Q(x)$ 是我们可以测得的预测分布。

对于 $P(x)$ 而言，其信息熵为

$$
H(P) = -\int P(x){\rm log}P(x)
$$

对于 $Q(x)$ 而言，由于其样本来自真实分布 $P(x)$，因此其信息熵中的概率依然为 $P(X)$，则其信息熵为

$$
H(P,Q) = -\int P(x){\rm log}Q(x)
$$

为了衡量二者的差异，定义 **相对熵** 为

$$
\begin{aligned}
KL(P\Vert Q) &= -\int P(x){\rm log}Q(x) - [-\int P(x){\rm log} P(x)]\\
&= - \int P(x){\rm log}\frac{Q(x)}{P(x)}
\end{aligned}
$$

相对熵又被称为 **KL 散度**。

- 由于 $P(x)$ 和 $Q(x)$ 在公式中的地位不是相等的，所以 $KL(P\Vert Q)\not\equiv KL(Q\Vert P)$。

- 当，$P(x) = Q(x)$时，$KL(P\Vert Q) = 0$。若 $P(x)$ 和 $Q(x)$ 有差异，$KL(P\Vert Q) > 0$。

KL 散度的非负性证明，利用了负指数函数是严格凸函数的性质，证明过程如下：

> 首先，$-log()$ 函数是上凹函数，如下面的示意图所示
> 根据 Jensen 不等式：$f(\mathbb E(x))\leq \mathbb E(f(x))$
> 根据含隐随机变量的 Jensen 不等式：$f(\mathbb E[\xi(z)])\leq \mathbb E[f(\xi(z))]$
> 令 $f(\cdot) = -log(\cdot)$，令 $\xi(z) = \frac{Q(x)}{P(x)}$，带入上式得
> $$
> \begin{aligned}
> \mathbb E_x[f(\xi(x))] &\geq f(\mathbb E_x[\xi (x)])\\
> \mathbb E_x[-log(\frac{Q(x)}{P(x)})] &\geq -log(E_x[\frac{Q(x)}{P(x)}]))\\
> -\int log(\frac{Q(x)}{P(x)}) P(x){\rm d}x &\geq -log(\int \frac{Q(x)}{P(x)}P(x){\rm d}x)\\
> -\int log(\frac{Q(x)}{P(x)}) P(x){\rm d}x &\geq -log(\int Q(x){\rm d}x)=-log(1)=0\\
> \end{aligned}
> $$

## 3.4. 交叉熵

遍地胡说. [详解机器学习中的熵、条件熵、相对熵和交叉熵](https://www.cnblogs.com/kyrieng/p/8694705.html)

KL 散度可以用来评估两个分布的差异程度，假设未知真实分布为 $P(x)$，已知估计分布为 $Q(x)$，根据其定义

$$
\begin{aligned}
KL(P\Vert Q) &= -\int P(x){\rm log}Q(x) - [-\int P(x){\rm log} P(x)]\\
&= H(P,Q) - H(P)
\end{aligned}
$$

注意到，后一项 $H(P)$ 是未知真实分布的信息熵，对于一个确定的未知分布而言是个常数。因此，在实际应用中我们只关心前一项的取值，前一项 $H(P,Q)$ 我们称为 **交叉熵**。

因此，我们可以得到如下的等式

$$
相对熵（KL散度） = 交叉熵 - 信息熵
$$

根据前面 $KL 散度\geq 0$ 的性质，我们可以知道，$交叉熵 \geq 信息熵$。

---

在机器学习中，我们希望在训练数据上模型学到的分布 $Q(x)$ 和真实数据的分布  $P(x)$ 越接近越好，所以我们可以使其相对熵最小。但是我们没有真实数据的分布，所以只能希望模型学到的分布 $Q(x)$ 和训练数据的分布 $P_t(x)$ 尽量相同。

假设训练数据是从总体中独立同分布采样的，那么我们可以通过最小化训练数据的经验误差来降低模型的泛化误差。即：

- **希望**学到的模型的分布和真实分布一致，$Q(x) \simeq P(x)$
- 但是真实分布不可知，假设训练数据是从真实数据中独立同分布采样的，$P_t(x) \simeq P(x)$
- 因此，我们**希望**学到的模型分布至少和训练数据的分布一致，$Q(x) \simeq P_t(x)$

根据之前的描述，最小化训练数据上的分布 $P_t(x)$ 与最小化模型分布 $Q(x)$ 的差异等价于最小化**相对熵**，即 ${\rm minimize}\ [KL(P_t(x)\vert \vert Q(x))]$。此时，$P_t(x)$ 就是 $KL(p\vert \vert q)$ 中的 $p$，即真实分布，$Q(x)$ 就是 $q$。又因为训练数据的分布 $p$ 是给定的，所以求 $KL(p\vert \vert q)$ 等价于求 $H(p,q)$。得证，**交叉熵可以用来计算学习模型分布与训练分布之间的差异**。

实际上，由于 $P(x)$ 是已知

$$
{\rm minimize}\ H(p,q) = {\rm minimize}\ [-\int P(x){\rm log}Q(x)] = {\rm minimize}\ \mathbb E_p[-log Q(x)]
$$

**任何一个负对数似然组成的损失都是定义在训练集上的经验分布和定义在模型上的真实分布之间的交叉熵。**

> ccj_zj. [多分类问题中的交叉熵](https://blog.csdn.net/ccj_ok/article/details/78066619)
> 
> **交叉熵是直接衡量两个分布，或者说两个model之间的差异。而似然函数则是解释以model的输出为参数的某分布模型对样本集的解释程度。因此，可以说这两者是“同貌不同源”，但是“殊途同归”啦。**

## 3.5. softmax 函数

**$n$ 分类问题** 的 softmax 函数定义如下

$$
\hat y_i = P(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
$$

假设某个训练样本经过神经网络后到达 softmax 层之前的输出向量为 $z=[ 1, 5, 3 ]$, 那么经过 softmax 函数后的概率分别为 $\hat y=[0.015,0.866,0.117]$。

假设期望输出标签为 $y = [0,1,0]$，那么交叉熵损失函数可以定义为

$$
L(\hat y, y) = -\sum_{j=1}^3 y_j {\rm ln} \hat{y}_j
$$

由于在分类问题中，期望输出标签是 one-hot 型变量。不失一般性，假设第 $j$ 个分量为 1 ，则损失函数为

$$
L(\hat y, y) = - {\rm ln} \hat{y}_j\ (y_j = 1)
$$

进行梯度下降时，根据链式法则

$$
\frac{\partial L}{\partial \omega} = \frac{\partial L}{\partial \hat y_j}\frac{\partial \hat y_j}{\partial z_i}\frac{\partial z_i}{\partial \omega_k}
$$

其中

$$
\frac{\partial L}{\partial \hat y_j} = -\frac{1}{\hat y_j}
$$

而 $\frac{\partial z_i}{\partial \omega_k}$ 根据网络具体形式，一般比较好求。

因此，重点在于求解中间的偏导项，需要分情况讨论

$j=i$ 时，表明反向传播至同样下标的上一层节点：

$$
\begin{aligned}
{\rm if}\quad j&=i:\\
\frac{\partial \hat y_j}{\partial z_i}&=\frac{\partial \hat y_i}{\partial z_i}\\
&=\frac{\partial }{\partial z_i}(\frac{e^{z_i}}{\sum_{k} e^{z_k}})\\
&=\frac{(e^{z_i})'\cdot \sum_{k} e^{z_k}-e^{z_i}\cdot e^{z_i}}{(\sum_{k} e^{z_k})^2}\quad(分式函数求导法则)\\
&=\frac{e^{z_i}}{\sum_{k} e^{z_k}} - \frac{e^{z_i}}{\sum_{k} e^{z_k}}\frac{e^{z_i}}{\sum_{k} e^{z_k}}\\
&= \hat y_j(1-\hat y_j)
\end{aligned}
$$

此时

$$
\frac{\partial L}{\partial z_j} = -\frac{1}{\hat y_j} \cdot \hat y_j(1-\hat y_j) = \hat y_j - 1
$$

可以看出形式非常简单，只要正向求一次得出结果，然后反向传梯度的时候，将结果减 1 即可。

$j\neq i$ 时，表明反向传播至不同下标的上一层节点：

$$
\begin{aligned}
{\rm if}\quad j&\neq i:\\
\frac{\partial \hat y_j}{\partial z_i}&=\frac{\partial }{\partial z_i}(\frac{e^{z_j}}{\sum_{k} e^{z_k}})\\
&=\frac{ {\rm d} e^{z_j}/{\rm d} e^{z_i}\cdot \sum_{k} e^{z_k}-e^{z_j}\cdot e^{z_i} }{(\sum_{k} e^{z_k})^2}\\
&=\frac{0\cdot \sum_{k} e^{z_k}-e^{z_j}\cdot e^{z_i}}{(\sum_{k} e^{z_k})^2}\\
&=-\frac{e^{z_j}}{\sum_{k} e^{z_k}}\frac{e^{z_i}}{\sum_{k} e^{z_k}}\\
&= -\hat y_j\hat y_i
\end{aligned}
$$

此时

$$
\frac{\partial L}{\partial z_j} = -\frac{1}{\hat y_j} \cdot (-\hat y_j\hat y_i) = \hat y_i
$$

形式同样非常简单，只要正向求一次得出结果，然后反向传梯度的时候，将它结果保存即可。

还是上面的例子，假设输出向量为 $z=[ 1, 5, 3 ]$, 那么经过 softmax 函数后的概率分别为 $\hat y=[0.015,0.866,0.117]$，交叉熵损失函数对 $z$ 的篇导数为 $\hat y'=[0.015,0.866-1,0.117] = [0.015,-0.134,0.117]$。

可以看出，softmax 配合 交叉熵损失函数 可以使得梯度下降非常简单。

# 4. 参考文献

[1] 产品经理马忠信. [应该如何理解概率分布函数和概率密度函数？](https://www.jianshu.com/p/b570b1ba92bb)

[2] 忆臻. [一文搞懂极大似然估计](https://zhuanlan.zhihu.com/p/26614750)

[3] 马同学. [如何理解贝叶斯推断和beta分布？](https://www.matongxue.com/madocs/910)

[4] 遍地胡说. [详解机器学习中的熵、条件熵、相对熵和交叉熵](https://www.cnblogs.com/kyrieng/p/8694705.html)

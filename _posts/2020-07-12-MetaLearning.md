---
layout: post
title:  "元学习文章阅读（MAML,Reptile）"
date:   2020-07-13 14:35:19
categories: Reading
tags: ML

---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# 目录

* [目录](#目录)
* [MAML](#MAML)
  * [算法](#算法)
  * [关于二重梯度的进一步解释](#关于二重梯度的进一步解释)
  * [FOMAML一阶近似简化](#FOMAML一阶近似简化)
  * [缺点](#缺点)
* [Reptile](#Reptile)
  * [算法](#算法)
  * [分析](#分析)
* [各类算法实现](#各类算法实现)
* [参考文献](#参考文献)

# MAML

2017.《Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks》

> The key idea underlying our method is to **train the model’s initial parameters** such that the model has maximal performance on a new task after the parameters have been updated through one or more gradient steps computed with a small amount of data from that new task.

本文的设想是**训练一组初始化参数**，通过在初始参数的基础上进行一或多步的梯度调整，来达到**仅用少量数据就能快速适应新task**的目的。为了达到这一目的，训练模型需要最大化新task的loss function的参数敏感度（*maximizing the sensitivity of the loss functions of new tasks with respect to the parameters*），当敏感度提高时，极小的参数（参数量）变化也可以对模型带来较大的改进。本文提出的算法可以适用于多个领域，包括少样本的回归、图像分类，以及增强学习，并且使用更少的参数量达到了当时（2017年）最先进的专注于少样本分类领域的网络的准确率。

核心算法示意图如下

![](..\assets\img\postsimg\20200713\3.jpg)

如上图所示，作者便将目标设定为，通过梯度迭代，找到对于task敏感的参数 $\theta$ 。训练完成后的模型具有对新task的学习域分布最敏感的参数，因此可以在仅一或多次的梯度迭代中获得最符合新任务的  $\theta^*$  ，达到较高的准确率。

## 算法

假设这样一个监督分类场景，目的是训练一个数学模型 $M_{fine-tune} $ ，对未知标签的图片做分类，则两大步骤如下：

1. 利用某一批数据集训练元模型 $M_{meta} $ 
2. 在另外一批数据集上精调（fine-tune）得到最终的模型 $M_{fine-tune} $ 。

MAML在监督分类中的算法伪代码如下：

![](..\assets\img\postsimg\20200713\6.jpg)

下面进行详细分析。该算法是 meta-train 阶段，目的是得到 $M_{meta} $；

**第一个Require**，反复随机抽取 $ \mathcal T$，得到一批（e.g. 1000个）task池，作为训练集 $ p(\mathcal T)$。假设一个 $ \mathcal T$ 包含5类，每类20个样本，随机选5样本作为support set（$D_i$），剩余15样本为query set（$D_i'$）。

> 训练样本就这么多，要组合形成那么多的task，岂不是不同task之间会存在样本的重复？或者某些task的query set会成为其他task的support set？没错！就是这样！我们要记住，MAML的目的，在于fast adaptation，即通过对大量task的学习，获得足够强的泛化能力，从而面对新的、从未见过的task时，通过fine-tune就可以快速拟合。task之间，只要存在一定的差异即可。

**第二个Require**，step size就是学习率，MAML是基于二重梯度（gradient by gradient），每次迭代有两次参数更新过程，所以有两个学习率可以调整。

**1**：随机初始化模型参数 $\theta$；

**2**：循环，对于每个epoch，进行若干batch；

**3**：随机选取若干个（比如4个） $ \mathcal T$  形成一个batch；

**4**：对于每个batch中的第 $i$ 个 $ \mathcal T$ ，进行**第一次**梯度更新*「inner-loop 内层循环」*。

**5**：选取 $ \mathcal T_i$ 中的 **support set**，共  $N\cdot K$个样本（5-way 5-shot=25个样本）

**6**：计算每个参数的梯度。原文写对每一个类下的 $K$ 个样本做计算。实际上参与计算的总计有 $N\cdot K$ 个样本。这里的loss计算方法，在回归问题中就是MSE；在分类问题中就是cross-entropy；

**7**：进行第一次梯度更新得到 $\theta'$，可以理解为对 $ \mathcal T_i$ 复制一个原模型 $f(\theta)$ 来更新参数；

**8**：挑出训练集中的 query set 数据用于后续二次梯度更新；

**9**：完成第一次梯度更新。

**10**：进行**第二次**梯度更新，此时计算出的梯度直接通过GD作用于原模型上，用于更新其参数。*「outer-loop 外层循环」*

大致与步骤 **7** 相同，但是不同点有三处：

- 隐含了**二重梯度**，需要计算 $\mathcal L_{T_i}f(\theta')$ 对 $\theta$ 的导数，而 $\mathcal L_{T_i}f(\theta')$ 是 $\theta'$ 的函数， $\theta'$ 又是 $\theta$ 的函数（见步骤**7**）；

- 不再分别利用每个 $ \mathcal T$ 的loss更新梯度，而是计算一个 batch 中模型 $L_{T_i}f(\theta')$ 的 loss 总和进行梯度下降；

- 参与计算的样本是task中的 **query set**（5way*15=75个样本），目的是增强模型在task上的泛化能力，避免过拟合support set。

**11**：结束在该batch中的训练，回到步骤**3**，继续采样下一个batch。

**总结**：MAML使用训练集优化内层循环，使用测试集优化模型，也就是外层循环。外层循环需要计算**二重梯度（gradient by gradient）**。

## 关于二重梯度的进一步解释

设初始化的参数为 $\theta$ ，每个任务的模型一开始的参数都是 $\theta$。

模型在 $\mathcal T_i$ 上经过训练后，参数就会变成 $\theta_i'$ ，用 $l_i(\theta_i')$ 表示模型在**每个任务 $\mathcal T_i$ 上的损失**。那么，对于这个Meta Learning而言，**整体的损失**函数应该是 $\theta$ 的函数：

$$
\mathcal L( \theta ) = \sum_{i=1}^N l_i(\theta_i')
$$

一旦我们可以计算 $\theta$ 的梯度，就可以直接更新 $\theta$ ：

$$
\theta' \leftarrow \theta-\eta\nabla_{\theta}\mathcal L(\theta)
$$

而所谓的假设即是：**每次训练只进行一次梯度下降**。这个假设听起来不可思议，但是却也有一定的道理，首先我们只是在Meta Learning的过程中只进行一次参数下降，而真正学习到了很好的 $\theta$ 之后自然可以进行多次梯度下降。只考虑一次梯度下将的原因有：

- Meta Learning会**快**很多；
- 如果能让模型**只经过一次**梯度下降就性能优秀，当然很好；
- Few-shot learning的数据有限，多次梯度下降很容易**过拟合**；
- 刚才说的可以在实际应用中**多次梯度下降**。

如果只经历了一次梯度下降，模型最后的参数就会变成：

$$
\theta' = \theta-\epsilon \nabla_{\theta}\mathcal l(\theta)
$$

对于每个不同任务，准确来说应该是：

$$
\theta_i' = \theta-\epsilon \nabla_{\theta}\mathcal l_i(\theta)
$$

下一步就是计算 $\theta$ 关于 $\mathcal L$ 的梯度。我们有：

$$
\nabla_\theta \mathcal L( \theta ) = \nabla_\theta \sum_{i=1}^N l_i(\theta_i') = \sum_{i=1}^N \nabla_\theta l_i(\theta_i')
$$

现在的问题是如何求 $\nabla_\theta l_i(\theta_i')$ ，略去下标 $i$，有：

$$
\begin{aligned}
\nabla_\theta l(\theta ') =
\left[   
    \begin{array}
    {}
    \partial l(\theta')/\partial \theta_1\\
	\partial l(\theta')/\partial \theta_2\\
	\vdots\\
	\partial l(\theta')/\partial \theta_n
	\end{array}
\right]
\end{aligned}
$$

注意 $l$ 是的 $\theta'$ 的函数，而 $\theta'$ 又和每一个 $\theta_i$ 有关，因此有：

$$
\frac {\partial l(\theta')}{\partial \theta_i} = \sum_j\frac{\partial l(\theta')}{\partial \theta_j'}\frac{\partial \theta_j'}{\partial \theta_i}
$$

也就是说，每一个 $\theta_i$ 通过影响不同的 $\theta_j '$，从而影响到了 $l$。$l$ 和 $\theta$ 的关系是很直接的，我们可以直接求$\frac{\partial l(\theta ')}{\partial \theta_j'}$ ，现在的问题是怎么求 $\frac{\partial \theta_j'}{\partial \theta_i}$。

注意到 $\theta'$ 和 $\theta$ 的关系也是显然的：

$$
\theta' = \theta-\epsilon \nabla_{\theta}\mathcal l(\theta)
$$

当 $i \neq j$ 时

$$
\frac{\partial \theta_j'}{\partial \theta_i} = 
- \epsilon\frac{\partial l^2(\theta)}{\partial \theta_i\partial \theta_j}
$$

当 $i = j$ 时

$$
\frac{\partial \theta_j'}{\partial \theta_i} = 
1 - \epsilon\frac{\partial l^2(\theta)}{\partial \theta_i\partial \theta_i}
$$

到此为止已经把梯度计算出来了，二重梯度也是MAML计算中最为耗时的部分。

## FOMAML一阶近似简化

在MAML的论文中提到了一种简化，它通过计算一重梯度来近似二重梯度。具体而言，假设学习率 $\epsilon \rightarrow 0^+$，则更新一次后的参数 $\theta'$ 对初始参数 $\theta$ 求偏导可变为

$$
(i \neq j) \; \frac{\partial \theta_j'}{\partial \theta_i} = 
- \epsilon\frac{\partial l^2(\theta)}{\partial \theta_i\partial \theta_j} \approx 0 \\
(i = j) \; \frac{\partial \theta_j'}{\partial \theta_i} = 1 - \epsilon\frac{\partial l^2(\theta)}{\partial \theta_i\partial \theta_i} \approx 1
$$

也就是说，可**将更新后的模型参数对模型原始参数的导数近似看作常数**（0或1）。

那么原来的偏导可近似为：

$$
\frac {\partial l(\theta')}{\partial \theta_i} = \sum_j\frac{\partial l(\theta')}{\partial \theta_j'}\frac{\partial \theta_j'}{\partial \theta_i} \approx
\frac {\partial l(\theta')}{\partial \theta_i'}
$$

整个梯度就可以近似为：

$$
\nabla_\theta l(\theta') \approx \nabla_{\theta'} l(\theta')
$$

简化后的一阶近似的MAML模型参数更新式为：

$$
\theta \leftarrow \theta - \alpha \nabla_{\theta'} \sum \mathcal L(\theta')\\
\theta_i' \leftarrow \theta - \beta \nabla_\theta l_i(\theta)\\
$$

一阶近似的MAML可以看作是如下形式的参数更新：假设每个batch只有一个task，某次采用第m个task来更新模型参数，得到$\hat\theta^m$，再求一次梯度来更新模型的原始参数$\phi$，将其从 $\phi^0$ 更新至 $\phi^1$，以此类推。

![](..\assets\img\postsimg\20200713\4.jpg)

与之相比，右边是模型预训练方法，它是将参数根据每次的训练任务一阶导数的方向来更新参数。

## 缺点

MAML的缺点[[2](#ref2)]：

1. Hard to train：paper中给出的backbone是4层的conv加1层linear，试想，如果我们换成16层的VGG，每个task在算fast parameter的时候需要计算的Hessian矩阵将会变得非常大。那么你每一次迭代就需要很久，想要最后的model收敛就要更久。

2. Robustness一般：不是说MAML的robustness不好，因为也是由一阶online的优化方法SGD求解出来的，会相对找到一个flatten minima location。然而这和非gradient-based meta learning方法求解出来的model的robust肯定是没法比的。

# Reptile

Reptile是OpenAI提出的一种非常简单的meta learning 算法。与MAML类似，也是学习网络参数的初始值。

## 算法

算法伪代码如下

![image-20200717113006074](..\assets\img\postsimg\20200713\7.jpg)

其中，$\phi$ 是模型的初始参数，$\tau$ 是某个 task，$SGD(L,\phi,k)$ 表示从$\phi$ 开始对损失函数$L$进行$k$次随机梯度下降，返回更新后的参数$W$。

在最后一步中，通过 $W-\phi$ 这种残差形式来更新一次初始参数。

算法当然也可设计为batch模式，如下

![image-20200717115435570](..\assets\img\postsimg\20200713\8.jpg)

如果k=1，该算法等价于「联合训练」（joint training，通过训练来最小化在一系列训练任务上期望损失）。

Reptile 要求 k>1，更新依赖于损失函数的高阶导数，k>1 时 Reptile 的行为与 k=1（联合训练）时截然不同。

Reptile与FOMAML紧密相关，但是与FOMAML不同，Reptile**无需对每一个任务进行训练-测试（training-testing）划分**。

相比MAML需要进行二重梯度计算，Reptile只需要进行一重梯度计算，计算速度更快。

Reptile的图例如下。

![](..\assets\img\postsimg\20200713\9.jpg)

## 分析

为什么 Reptile 有效？首先以两步 SGD 为例分析参数更新过程

$$
\phi_0 = \phi\\
\phi_1 = \phi_0 - \alpha L_0'(\phi_0)\\
\phi_2 = \phi_0 - \alpha L_0'(\phi_0) - \alpha L_1'(\phi_1)
$$

下面定义几个**辅助变量**

$$
g_i = L_i'(\phi_i)
\;\;(gradient \; obtained\; during\;SGD)\\
\phi_{i+1} = \phi_i-\alpha g_i
\;\;(sequence\;of\;parameters)\\
\overline{g}_i = L_i'(\phi_0)
\;\;(gradient\;at\;initial\;point)\\
\overline{H}_i = L_i''(\phi_0)
\;\;(Hessian\;at\;initial\;point)\\
$$

采用**泰勒展开**对 $g_i$ 展开至二阶导加高次项的形式

$$
g_i = L_i'(\phi_i) = L_i'(\phi_0) + L_i''(\phi_0)(\phi_i - \phi_0) + O(||\phi_i - \phi_0||^2)\\
= \overline{g}_i + \overline{H}_i(\phi_i - \phi_0) + O(\alpha^2)\\
= \overline{g}_i - \alpha\overline{H}_i\sum_0^{i-1}g_i + O(\alpha^2)\\
= \overline{g}_i - \alpha\overline{H}_i\sum_0^{i-1}\overline{g}_i + O(\alpha^2)\\
$$

最后一步的依据是，$g_i = \overline{g}_i + O(\alpha)$ 带入倒数第二行时，后面的 $O(\alpha)$ 与求和符号前的 $\alpha$ 相乘，即变为 $O(\alpha^2)$ 从而合并为一项。

下面取 **k=2** ，即两步，分别推导 MAML、FOMAML、Reptile 的梯度。[这个参考]([https://www.cnblogs.com/veagau/p/11816163.html#42-reptile%E5%AE%9E%E7%8E%B0](https://www.cnblogs.com/veagau/p/11816163.html#42-reptile实现)) [[3](#ref3)] 考虑了k为任意情况下的通解，在k=2时的情况与本文相同。

对于二阶的MAML，初始参数 $\phi_0$ 首先在support set上梯度更新一次得到 $\phi_1$ ，然后将 $\phi_1$ 在 query set 上计算损失函数，再计算梯度更新模型的初始参数。即 query set 的 loss 要对 $\phi_0$ 求导，链式法则 loss 对  $\phi_1$ 求导乘以  $\phi_1$ 对 $\phi_0$ 求导

$$
g_{MAML} = \frac{\partial}{\partial\phi_0}L_1(\phi_1) = \frac{\partial \phi_1}{\partial \phi_0} L_1'(\phi_1) \\
 = (I-\alpha L_0''(\phi_0))L_1'(\phi_1)\\
 = (I-\alpha L_0''(\phi_0))(L_1'(\phi_0) + L_1''(\phi_0)(\phi_1 - \phi_0) + O(\alpha^2))\\
 = (I-\alpha L_0''(\phi_0))(L_1'(\phi_0) + L_1''(\phi_0)(\phi_1 - \phi_0)) + O(\alpha^2)\\
 = (I-\alpha L_0''(\phi_0))(L_1'(\phi_0) - \alpha L_1''(\phi_0)L_0'(\phi_0)) + O(\alpha^2)\\
 = L_1'(\phi_0)-\alpha L_0''(\phi_0)L_1'(\phi_0) - \alpha L_1''(\phi_0)L_0'(\phi_0) + O(\alpha^2)\\
$$

对于FOMAML，其一阶简化是简化了参数 $\phi_1$ 对初始参数 $\phi_0$ 求导部分，即 $\frac{\partial \phi_i}{\partial \phi_0} = const$，参见2.3节。则只剩下 loss 对参数 $\phi_1$ 的求导。

$$
g_{FOMAML} = L_1'(\phi_i) = L_1'(\phi_0) + L_1''(\phi_0)(\phi_1 - \phi_0) + O(\alpha^2)\\
= L_1'(\phi_0) -\alpha L_1''(\phi_0)L_0'(\phi_0) + O(\alpha^2)
$$

对于Reptile，根据梯度定义和SGD过程有

$$
g_{Reptile} = (\phi_0 - \phi_2)/\alpha = L_0'(\phi_0)+L_1'(\phi_1)\\
= L_0'(\phi_0)+L_1'(\phi_0)-\alpha L_0''(\phi_1)L_0'(\phi_0) + O(\alpha^2)
$$

接下来对上面的三个梯度进行变量替换，全部用之前定义的辅助变量来表示

$$
g_{MAML} = g_1 - \alpha \overline{H_0}\overline{g}_1-\alpha\overline{H_1}\overline{g}_0+O(\alpha^2)\\
g_{FOMAML} = g_1 = \overline{g}_1-\alpha\overline{H}_1\overline{g}_0+O(\alpha^2)\\
g_{Reptile} = g_0+g_1 = \overline{g}_0+\overline{g}_1-\alpha \overline{H}_1\overline{g}_0 + O(\alpha^2)\\
$$

再次定义两个期望参数如下。

**第一个**：AvgGrad 定义为loss对初始参数的梯度的平均期望

$$
AvgGrad = \mathbb E_{\tau,0}[\overline{g}_0] =\mathbb E_{\tau,1}[\overline{g}_1]
$$

（-AvgGrad）是参数 $\phi$ 在最小化 joint training 问题中的下降方向。就是想在所有batch上减小loss，也就是减小整体的任务损失。

为什么此处 $\overline{g}_0$ 和 $\overline{g}_1$ 的期望相等呢，因为它们都是表示的是loss对原始参数的梯度，只不过对应于不同的batch。在minibatch中，一个batch用于进行一次梯度下降，因为batch是随机的，所以loss对原始参数的梯度与batch是无关的（？）。

**第二个**：AvgGradInner

$$
AvgGradInner = \mathbb E_{\tau,0,1}[\overline{H}_0\overline{g}_1]
= \mathbb E_{\tau,0,1}[\overline{H}_1\overline{g}_0]
= \frac{1}{2}\mathbb E_{\tau,0,1}[\overline{H}_0\overline{g}_1+\overline{H}_1\overline{g}_0]
= \frac{1}{2}\mathbb E_{\tau,0,1}[\frac{\partial}{\partial \phi_0}(\overline{g}_0\overline{g}_1)]
$$

(-AvgGradInner) 的方向可以增大不同minibatch间梯度的内积，从而提高泛化能力。换句话说，AvgGradInner是 $\overline{g}_0\overline{g}_1$ 的对原始参数的导数，因为梯度在参数更新时是加负号的，**所以是在最大化同一任务中不同minibatch之间梯度的内积**。对其中一个batch进行梯度更新会显著改善另一个batch的的表现，这样就增加了模型的泛化性和快速学习的能力。

下面就可以对上述三个梯度进行进一步替换

$$
\mathbb{E}[g_{MAML}] = (1)AvgGrad - (2\alpha)AvgGradInner + O(\alpha^2)\\
\mathbb{E}[g_{FOMAML}] = (1)AvgGrad - (\alpha)AvgGradInner + O(\alpha^2)\\
\mathbb{E}[g_{Reptile}] = (2)AvgGrad - (\alpha)AvgGradInner + O(\alpha^2)\\
$$

扩展到 k>2 的情况有 [[3](#ref3)] 

$$
\mathbb{E}[g_{MAML}] = (1)AvgGrad - (2(k-1)\alpha)AvgGradInner + O(\alpha^2)\\
\mathbb{E}[g_{FOMAML}] = (1)AvgGrad - ((k-1)\alpha)AvgGradInner + O(\alpha^2)\\
\mathbb{E}[g_{Reptile}] = (2)AvgGrad - (\frac{1}{2}k(k-1)\alpha)AvgGradInner + O(\alpha^2)\\
$$

另一种分析有效的方法借助了流形，Reptile 收敛于一个解，这个解在欧式空间上与每个任务的最优解的流形接近。没看懂不管了。

## 实验

**少样本分类**

![img](..\assets\img\postsimg\20200713\11.jpg)

![img](..\assets\img\postsimg\20200713\12.jpg)

从两个表格中的数据可以看出，MAML与Reptile在加入了转导（Transduction）后，在Mini-ImageNet上进行实验，Reptile的表现要更好一些，而Omniglot数据集上正好相反。

**不同的内循环梯度组合比较**

通过在内循环中使用四个不重合的Mini-Batch，产生梯度数据 g1,g2,g3,g4 ，然后将它们以不同的方式进行线性组合（等价于执行多次梯度更新）用于外部循环的更新，进而比较它们之间的性能表现，实验结果如下图：

![img](..\assets\img\postsimg\20200713\13.jpg)

从曲线可以看出：

- 仅使用一个批次的数据产生的梯度的效果并不显著，因为相当于让模型用见到过的少量的数据去优化所有任务。
- 进行了两步更新的Reptile（绿线）的效果要明显不如进行了两步更新的FOMAML（红线），因为Reptile在AvgGradInner上的**权重**（AvgGradInner与AvgGrad的系数的比例）要小于FOMAML。
- 随着mini-batch数量的增多，所有算法的性能也在提升。通过同时利用多步的梯度更新，Reptile的表现要比仅使用最后一步梯度更新的FOMAML的表现好。

**内循环中Mini-Batch 重合比较**

Reptile和FOMAML在内循环过程中都是使用的SGD进行的优化，在这个优化过程中任何微小的变化都将导致最终模型性能的巨大变化，因此这部分的实验主要是探究两者对于内循环中的超数的敏感性，同时也验证了FOMAML在minibatch以错误的方式选取时会出现显著的性能下降情况。

mini-batch的选择有两种方式：

- **shared-tail（共尾）**：最后一个内循环的数据来自以前内循环批次的数据
- **separate-tail（分尾）**：最后一个内循环的数据与以前内循环批次的数据不同

![img](..\assets\img\postsimg\20200713\14.jpg)

采用不同的mini-batch选取方式在FOMAML上进行实验，发现随着内循环迭代次数的增多，采用分尾方式的FOMAML模型的测试准确率要高一些，因为在这种情况下，测试的数据选取方式与训练过程中的数据选取方式更为接近。

当采用不同的批次大小时，采用共尾方式选取数据的FOMAML的准确性会随着批次大小的增加而显著减小。当采用**full-batch**时，共尾FOMAML的表现会随着外循环步长的加大而变差。

**共尾FOMAML的表现如此敏感的原因可能是最初的几次SGD更新让模型达到了局部最优，以后的梯度更新就会使参数在这个局部最优附近波动。**

# 比较

再次比较Model Pre-training、MAML 和 Reptile方法。

![](..\assets\img\postsimg\20200713\4.jpg)

（左：MAML。右：Model Pre-training）

![](..\assets\img\postsimg\20200713\9.jpg)

（Reptile）

Pre-training采用task的loss对参数的梯度更新模型的原始参数。

MAML采用task的第二次梯度计算更新模型的原始参数。

Reptile采用多次梯度计算更新模型的原始参数。

![10](..\assets\img\postsimg\20200713\10.jpg)

上面这个图不具体，但是很直观的展示了这些算法的区别。$g_i$ 表示第 $i$ 次负梯度计算。这里的MAML是一阶的，沿着 $g_2$ 方向更新，Reptile 沿着 $g_1+g_2$  的方向更新，而我们常规的预训练模型就是沿着 $g_1$ 方向更新。

# 各类算法实现

[dragen-1860 的 Pytorch 实现](https://github.com/dragen1860/MAML-Pytorch)：https://github.com/dragen1860/MAML-Pytorch

[Tensorflow实现](https://github.com/dragen1860/MAML-TensorFlow)：https://github.com/dragen1860/MAML-TensorFlow

[First-order近似实现Reptile](https://github.com/dragen1860/Reptile-Pytorch)：https://github.com/dragen1860/Reptile-Pytorch

# 参考文献

<span id="ref1">[1]</span>  [Rust-in](https://www.zhihu.com/people/rustinnnnn). [MAML 论文及代码阅读笔记](https://zhuanlan.zhihu.com/p/66926599).

<span id="ref2">[2]</span> 人工智障. [MAML算法，model-agnostic metalearnings?](https://www.zhihu.com/question/266497742/answer/550695031)

<span id="ref3">[3]</span> [Veagau](https://www.cnblogs.com/veagau/). [【笔记】Reptile-一阶元学习算法](https://www.cnblogs.com/veagau/p/11816163.html)

[4] [pure water](https://blog.csdn.net/qq_41694504). [Reptile原理以及代码详解](https://blog.csdn.net/qq_41694504/article/details/106750606)
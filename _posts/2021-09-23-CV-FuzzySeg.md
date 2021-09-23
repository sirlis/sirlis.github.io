---
title: 深度学习文章阅读（Image Segmentation）
date: 2021-09-23 09:53:49 +0800
categories: [Academic, Paper]
tags: [deeplearning]
math: true
---

本文介绍了模糊惩罚稀疏编码在扩散张量磁共振图像分割中的应用。Fuzziness Penalized Sparse Coding for Diffusion Tensor Magnetic Resonance Image Segmentation

<!--more-->

 ---

- [1. 参数定义](#1-参数定义)
- [2. 模糊惩罚稀疏编码](#2-模糊惩罚稀疏编码)
- [3. 优化](#3-优化)
  - [3.1. 参数更新](#31-参数更新)
  

# 1. 参数定义

- 扩散张量 $\boldsymbol{T}$：

$$
\boldsymbol{T}=[\boldsymbol v_1,\boldsymbol v_2,\boldsymbol v_3]
\begin{bmatrix}
e_1 & 0 &0 \\
0 & e_2 &0 \\
0 & 0 & e_3
\end{bmatrix}
[\boldsymbol v_1,\boldsymbol v_2,\boldsymbol v_3]^T
$$

其中

- $e_1\geq e_2 \geq e_3$ 是特征值；
- $\boldsymbol{v}_1, \boldsymbol{v}_2, \boldsymbol{v}_3$ 是特征向量；
- $\boldsymbol{v}_1=[v_{11}, v_{12}, v_{13}]$ 是主要扩散方向，即最大特征值 $e_1$ 对应的特征向量；

描述符：

- 几何描述符：

$$
\boldsymbol{x}_i^G=[e_1, e_2,e_3,FA, MD, RA, VR, RD, PA, SA, LA]\in \mathbb R^{11}
$$

- 方向描述符：

$$
\boldsymbol{x}_i^O=\frac{1}{\vert\vert\boldsymbol{v}_1\vert\vert} (v_{11}^2-v_{12}^2, 2v_{11}v_{12}, 2v_{11}v_{13}, 2v_{12}v_{13},\frac{1}{\sqrt{3}}(2v_{13}^2-v_{11}^2-v_{12}^2)) \in \mathbb R^{5}
$$

- 原始数据域下，核磁共振体素的特征向量：

$$
\boldsymbol{x}_i = [t_1, \boldsymbol x_i^G,\boldsymbol x_i^O]\in \mathbb R^{k\times 1}, \; k=17
$$

- 稀疏编码域下，体素特征向量的稀疏表示

$$
\boldsymbol s_i\in \mathbb R^{l\times 1}
$$

- 超完备词典

$$
\boldsymbol D = [\boldsymbol d_1, \boldsymbol d_2, \cdots, \boldsymbol d_l]\in \mathbb R^{k\times l}
$$

# 2. 模糊惩罚稀疏编码

输入训练样本 $\boldsymbol X = [\boldsymbol x_1, \cdots, \boldsymbol x_n]\in \mathbb R^{k\times n}$，对于每个样本可以训练字典得到稀疏编码

$$
F(\boldsymbol s, \boldsymbol D)=\min_{\boldsymbol s, \boldsymbol D} \sum_{i=1}^n(\frac{1}{2}\vert\vert\boldsymbol x_i - \boldsymbol D \boldsymbol s_i\vert\vert_2^2+\lambda\vert\vert \boldsymbol s_i\vert\vert_1),\\
s.t.\vert\vert\boldsymbol d_j\vert\vert_2^2 < 1 \; \forall j
$$

专家提供的硬标签为 $\boldsymbol l_i \in \mathbb R^C$（one-hot 形式）。软模糊隶属度 $\boldsymbol u_i=[u_1,\cdots, u_C] \in \mathbb R^C$（分量和为 1），表明样本 $\boldsymbol x_i$ 与多个其他类别的关系。

模糊隶属度采用模糊k近邻算法（FKNN）计算。首先计算样本特征向量间的欧式距离矩阵，然后对该距离矩阵进行排序，选取其中 $k$ 个最近邻的训练样本。对于属于第 $m$ 类别的样本 $\boldsymbol x$ ，其模糊隶属度计算方式为

$$
\begin{array}{l}
\hat u_{ij} = \left\{
\begin{aligned}
\gamma + (1-\gamma)(n_{jm}/k) & & j=m \\
(1-\gamma)(n_{jm}/k)  & & j\not=m \\
\end{aligned}
\right.
 \end{array}
$$

其中 $\gamma= \frac{N-C}{2^hN}$。N，C，h是啥？？

$n_{jm}$ 是属于第 $j$ 类别的邻居个数。$h \in (0,1),\gamma \in (0,1)$ 是用于控制模糊隶属度的常数。为了简化计算，进一步将模糊的分进行归一化，使得 $\sum_j \hat{u}_{ij} = 1$。

模糊度对应原始特征域的标注数据，采用模糊惩罚稀疏编码（FPSC）来在稀疏编码空间保留上述模糊度。

$$
G(\boldsymbol{s},\boldsymbol{u},\boldsymbol{D})= \min_{\boldsymbol{s},\boldsymbol{u},\boldsymbol{D}} [F(\boldsymbol{s},\boldsymbol{D}) + {\eta_1}\sum\limits_{i,j} {u}_{ij}\vert\vert\boldsymbol{s}_i - {\boldsymbol{c}_j}\vert\vert_2^2+
\eta_2\sum\limits_{i} I(i\in\Omega) \vert\vert \boldsymbol u_i-\hat{\boldsymbol u}_i\vert\vert_2^2],\\
s.t.\;\boldsymbol{u}_{i}\boldsymbol{1} = 1\;\forall i
$$

- 第一项：前面定义的稀疏编码模型；
- 第二项：稀疏编码表征 $\boldsymbol s_i$ 聚类到某些中心 $\boldsymbol c_j$；
- 第三项：模糊得分 $\boldsymbol u$ 被惩罚到与专家打分 $\hat{ \boldsymbol u}$ 一致。

其中，$\Omega$ 表示监督数据集，$I(i\in\Omega)$ 是一个指示函数，如果 $i\in\Omega$ 则取值为 1，否则为 0。


# 3. 优化

难点在于稀疏编码表征 $\boldsymbol s_i$ 的更新规则，因为他即在稀疏编码模型中出现，又在模糊聚类部分出现。参考稀疏优化的已有研究，引入辅助变量 $\boldsymbol t_i$ ，将优化转为如下形式

$$
\min_{\boldsymbol{s},\boldsymbol{u},\boldsymbol{D}}\;  G(\boldsymbol{s},\boldsymbol{u},\boldsymbol{D}), \; s.t. ~~\boldsymbol{s}_i=\boldsymbol{t}_i ~\forall i.
$$

上式是一个典型的约束优化问题，采用交替方向乘子法（ADM）可以将约束松弛到目标函数中

$$
\begin{aligned}
L(\boldsymbol{s},\boldsymbol{t},\boldsymbol{u},\boldsymbol{D}) &= \sum\limits_i {\lambda \vert\vert{\boldsymbol{t}_i}|{|_1} + \frac{1}
{2}\vert\vert{\boldsymbol{x}_i} - } \boldsymbol{D}{\boldsymbol{s}_i}\vert\vert_2^2 \\
& + {\eta _1}\sum\limits_{i,j} {u}_{ij}\vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2 \\
& + {\eta _2}\sum\limits_{i}I(i \in \Omega )\vert\vert{\hat{\boldsymbol{u}}_i} - {\boldsymbol{u}_i}\vert\vert_2^2 \\
& + < {\theta _{i}},\boldsymbol{u}_i\boldsymbol{1} - 1 >+
< {\mu _{i}},{\boldsymbol{t}_i} - {\boldsymbol{s}_i} >  \\
& + \frac{{{\rho}}}
{2}(\vert\vert{\boldsymbol{t}_i} - {\boldsymbol{s}_i}\vert\vert_2^2+\vert\vert\boldsymbol{u}_i\boldsymbol{1} - 1\vert\vert_2^2)
\end{aligned}
$$

其中，$\mu_i\in \mathbb R^{1\times l}, \theta_i\in \mathbb R^1, \; \forall i$ 是拉格朗日乘子，$\rho\in\mathbb R^1$ 是增广拉格朗日参数。上述形式是典型的ADM形式，可以采用两步法求解，第一步参数更新，第二部双升？

## 3.1. 参数更新

首先更新 $\boldsymbol s_i$，剔除与 $\boldsymbol s_i$ 无关的项，则有

$$
\begin{aligned}
\min_{\boldsymbol s_i} \; &  \frac{1}{2}\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2^2 + {\eta _1}\sum\limits_{j} u^{(k)}_{ij}\vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2 \\
& +  < {\mu_{i}},{\boldsymbol{t}^{(k)}_i} - {\boldsymbol{s}_i} >  + \frac{{{\rho}}}
{2}\vert\vert{\boldsymbol{t}^{(k)}_i} - {\boldsymbol{s}_i}\vert\vert_2^2\\
\end{aligned}
$$

对上式求偏导，第一项为

$$
\begin{aligned}
&\frac{\partial}{\partial \boldsymbol s_i} \frac{1}{2}\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2^2 \\
=&\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2 \cdot \frac{\boldsymbol{x}_i -  \boldsymbol{D}^{(k)}\boldsymbol{s}_i}{\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2 }\cdot \frac{\partial}{\partial \boldsymbol s_i}[{\boldsymbol{x}_i} - \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}] \quad (129)\\
=&({\boldsymbol{x}_i} - \boldsymbol{D}^{(k)}) \cdot (-{\boldsymbol D^{(k)}}^T) \quad (69)\\
=&{\boldsymbol D^{(k)}}^T\boldsymbol D^{(k)}
\end{aligned}
$$
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
    - [3.1.1. s 更新](#311-s-更新)
    - [3.1.2. t 更新](#312-t-更新)
    - [3.1.3. u 更新](#313-u-更新)
    - [3.1.4. D 更新](#314-d-更新)
  

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
s.t.\vert\vert\boldsymbol d_j\vert\vert_2^2 < 1 \; \forall j\in[1,\cdots,l]
$$

假设一共有 $C$ 类。

专家提供的硬标签为 $\boldsymbol l_i \in \mathbb R^C$（one-hot 形式）。软模糊隶属度 $\boldsymbol u_i=[u_1,\cdots, u_C] \in \mathbb R^C$（分量和为 1），表明样本 $\boldsymbol x_i$ 与多个其他类别的关系。其中，$C$ 为类别数。

模糊隶属度采用模糊k近邻算法（FKNN）计算。首先计算样本特征向量间的欧式距离矩阵，然后对该距离矩阵进行排序，选取其中 $k$ 个最近邻的训练样本。对于属于第 $m$ 类别的样本 $\boldsymbol x_i$，其模糊隶属度计算方式为

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

其中 $\gamma= \frac{N-C}{2^hN}$，$h,\lambda\in(0,1)$ 是控制模糊隶属度取值的常数，$n_{jm}$ 是属于第 $j$ 个类别的邻居样本个数。N，C是啥？？


$n_{jm}$ 是属于第 $j$ 类别的邻居个数。$h \in (0,1),\gamma \in (0,1)$ 是用于控制模糊隶属度的常数。为了简化计算，进一步将模糊的分进行归一化，使得 $\sum_j \hat{u}_{ij} = 1$。

> 进一步解读：对于每个样本，选取其 $k$ 个最近样本。计算该选择的样本对所有类别的模糊隶属度，该样本对应该类别的隶属度为第一个式子，该样本对于所有其它类别的隶属度为第二个式子；其中，每个隶属度右与对应类别的近邻样本个数有关。最后做一个归一化。


模糊度对应原始特征域的标注数据，采用模糊惩罚稀疏编码（FPSC）来在稀疏编码空间保留上述模糊度。

$$
G(\boldsymbol{s},\boldsymbol{u},\boldsymbol{D})= \min_{\boldsymbol{s},\boldsymbol{u},\boldsymbol{D}} [F(\boldsymbol{s},\boldsymbol{D}) + {\eta_1}\sum\limits_{i,j} {u}_{ij}\vert\vert\boldsymbol{s}_i - {\boldsymbol{c}_j}\vert\vert_2^2+
\eta_2\sum\limits_{i} I(i\in\Omega) \vert\vert \boldsymbol u_i-\hat{\boldsymbol u}_i\vert\vert_2^2],\\
s.t.\;\boldsymbol{u}_{i}\boldsymbol{1} = 1\;\forall i\in[1,\cdots,n]
$$

- 第一项：前面定义的稀疏编码模型；
- 第二项：稀疏编码表征 $\boldsymbol s_i$ 聚类到某些中心 $\boldsymbol c_j$；
- 第三项：模糊得分 $\boldsymbol u$ 被惩罚到与专家打分 $\hat{ \boldsymbol u}$ 一致。

其中，$\Omega$ 表示监督数据集，$I(i\in\Omega)$ 是一个指示函数，如果 $i\in\Omega$ 则取值为 1，否则为 0。

- $i=1,2,\cdots,n$ 遍历每个样本；
- $j=1,2,\cdots,C$ 遍历每个类别；


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
< {\boldsymbol \mu_i},{\boldsymbol{t}_i} - {\boldsymbol{s}_i} >  \\
& + \frac{\rho}
{2}(\vert\vert{\boldsymbol{t}_i} - {\boldsymbol{s}_i}\vert\vert_2^2+\vert\vert\boldsymbol{u}_i\boldsymbol{1} - 1\vert\vert_2^2)
\end{aligned}
$$

其中，$\boldsymbol \mu_i\in \mathbb R^{l\times 1}, \theta_i\in \mathbb R^1, \; \forall i$ 是拉格朗日乘子，$\rho\in\mathbb R^1$ 是增广拉格朗日参数。上述形式是典型的ADM形式，可以采用两步法求解，第一步参数更新，第二部双升？

## 3.1. 参数更新

### 3.1.1. s 更新

**注：后文中公式的括号内编号为 《The Matrix Cookbook》 书内对应参考公式编号。**

首先更新 $\boldsymbol s_i$，剔除与 $\boldsymbol s_i$ 无关的项，则有优化目标为

$$
\begin{aligned}
\min_{\boldsymbol s_i} \; &  \frac{1}{2}\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2^2\\
& + {\eta _1}\sum\limits_{j} u^{(k)}_{ij}\vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2 \\
& +  < {\mu_{i}},{\boldsymbol{t}^{(k)}_i} - {\boldsymbol{s}_i} > \\
& + \frac{\rho}
{2}\vert\vert{\boldsymbol{t}^{(k)}_i} - {\boldsymbol{s}_i}\vert\vert_2^2\\
\end{aligned}
$$

其中 $(k)$ 表示当前第 $k$ 步更新时的参数，涉及的参数为待更新的 $\boldsymbol s_i, \boldsymbol t_i, \boldsymbol D$。

对上式求偏导，第一项求偏导为

$$
\begin{aligned}
&\frac{\partial}{\partial \boldsymbol s_i} \frac{1}{2}\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2^2 \\
=&\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2 \cdot \frac{\boldsymbol{x}_i -  \boldsymbol{D}^{(k)}\boldsymbol{s}_i}{\vert\vert{\boldsymbol{x}_i} -  \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}\vert\vert_2 }\cdot \frac{\partial}{\partial \boldsymbol s_i}[{\boldsymbol{x}_i} - \boldsymbol{D}^{(k)}{\boldsymbol{s}_i}] \quad (129)\\
=&({\boldsymbol{x}_i} - \boldsymbol{D}^{(k)}\boldsymbol{s}_i) \cdot (-{\boldsymbol D^{(k)}}^T) \quad (69)\\
=&{\boldsymbol D^{(k)}}^T\boldsymbol D^{(k)}\boldsymbol{s}_i-{\boldsymbol D^{(k)}}^T\boldsymbol x_i\quad \in \mathbb R^{l\times 1}
\end{aligned}
$$

第二项求偏导为

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol s_i} {\eta _1}\sum\limits_{j} u^{(k)}_{ij}\vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2 
=&2\eta_1\sum\limits_{j} u^{(k)}_{ij}(\boldsymbol{s}_i - \boldsymbol{c}_j)\quad (129)\\
=&2\eta_1 \bar u \boldsymbol s_i - 2\eta_1\bar{\boldsymbol c}\\
where \quad &\bar u=\sum\limits_j u^{(k)}_{ij},\quad \bar{ \boldsymbol{c}}= \sum \limits_{j} u^{(k)}_{ij} \boldsymbol{c}_j
\end{aligned}
$$

第三项为矩阵内积，根据内积的定义有

$$
< \boldsymbol \mu_i,\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i > = tr(\boldsymbol \mu_i^T\cdot(\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i))
$$

则求偏导为

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol s_i}< \boldsymbol \mu_i,\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i >
=& \frac{\partial}{\partial \boldsymbol s_i}  tr(\boldsymbol \mu_i^T\cdot(\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i))\\
=&-\boldsymbol \mu_i\quad (101)
\end{aligned}
$$

或者根据向量内积定义（$\boldsymbol \mu_i \in\mathbb R^{l\times 1}$ 和 $\boldsymbol t_i-\boldsymbol s_i \in\mathbb R^{l\times 1}$ 都是向量），去掉 $tr$ 符号，则求偏导为

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol s_i}< \boldsymbol \mu_i,\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i >
=& \frac{\partial}{\partial \boldsymbol s_i}  (\boldsymbol \mu_i^T\cdot(\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i))\\
=&-\boldsymbol \mu_i \quad (70)
\end{aligned}
$$

第四项求偏导为

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol s_i} \frac{\rho}
{2}\vert\vert \boldsymbol{t}^{(k)}_i - {\boldsymbol{s}_i}\vert\vert_2^2
=&\rho\vert\vert \boldsymbol{t}^{(k)}_i - {\boldsymbol{s}_i}\vert\vert_2 \frac{\boldsymbol{t}^{(k)}_i - \boldsymbol{s}_i}{\vert\vert \boldsymbol{t}^{(k)}_i - {\boldsymbol{s}_i}\vert\vert_2}\cdot (-\boldsymbol I)\quad (129)\\
=&\rho(\boldsymbol{s}_i - \boldsymbol{t}^{(k)}_i)
\end{aligned}
$$

则求偏导后完整的表达式为

$$
\begin{aligned}
&\frac{\partial}{\partial \boldsymbol s_i} L(\boldsymbol{s}_i,\boldsymbol{t},\boldsymbol{u},\boldsymbol{D})\\
= &{\boldsymbol D^{(k)}}^T\boldsymbol D^{(k)}\boldsymbol{s}_i-{\boldsymbol D^{(k)}}^T\boldsymbol x_i + 2\eta_1 \bar u \boldsymbol s_i - 2\eta_1\bar{\boldsymbol c} - \boldsymbol \mu_i + \rho(\boldsymbol{s}_i - \boldsymbol{t}^{(k)}_i)\\
=& ({\boldsymbol D^{(k)}}^T\boldsymbol D^{(k)} + 2\eta_1 \bar u + \rho)\boldsymbol s_i - ({\boldsymbol D^{(k)}}^T\boldsymbol x_i+2\eta_1\bar{\boldsymbol c}+\rho\boldsymbol{t}^{(k)}_i+\boldsymbol \mu_i)
\end{aligned}
$$

令偏导数等于 0，则有

$$
\boldsymbol s_i = ({\boldsymbol D^{(k)}}^T\boldsymbol D^{(k)} + 2\eta_1 \bar u + \rho)^{-1}({\boldsymbol D^{(k)}}^T\boldsymbol x_i+2\eta_1\bar{\boldsymbol c}+\rho\boldsymbol{t}^{(k)}_i+\boldsymbol \mu_i)
$$

### 3.1.2. t 更新

剔除与 $\boldsymbol t_i$ 无关的项，则有优化目标为

$$
\begin{aligned}
\min_{\boldsymbol s_i} \; &   \sum\limits_i \lambda \vert\vert{\boldsymbol{t}_i}|{|_1}\\
& + < {\boldsymbol \mu_i},{\boldsymbol{t}_i} - {\boldsymbol{s}_i} >  \\
& + \frac{\rho}{2}\vert\vert{\boldsymbol{t}_i} - {\boldsymbol{s}_i}\vert\vert_2^2
\end{aligned}
$$

注意到，$\boldsymbol \mu_i \in\mathbb R^{l\times 1}$ 和 $\boldsymbol t_i-\boldsymbol s_i \in\mathbb R^{l\times 1}$ 都是向量，则内积项为

$$
< {\boldsymbol \mu_i},{\boldsymbol{t}_i} - {\boldsymbol{s}_i} > = {\boldsymbol \mu_i}^T({\boldsymbol{t}_i} - {\boldsymbol{s}_i})
$$

二范数项为

$$
\frac{\rho}{2}\vert\vert\boldsymbol{t}_i - \boldsymbol{s}_i\vert\vert_2^2 = \frac{\rho}{2}(\boldsymbol t_i - \boldsymbol s_i)^T(\boldsymbol t_i - \boldsymbol s_i)
$$

对于 $\boldsymbol t_i$ 而言，损失函数中的 $\boldsymbol \mu_i$ 没有影响，因此不妨将**损失函数后两项的和**配方成如下形式

$$
\begin{aligned}
&< {\boldsymbol \mu_i},{\boldsymbol{t}_i} - {\boldsymbol{s}_i} > + \frac{\rho}{2}\vert\vert\boldsymbol{t}_i - \boldsymbol{s}_i\vert\vert_2^2 \\
=& {\boldsymbol \mu_i}^T({\boldsymbol{t}_i} - {\boldsymbol{s}_i}) + \frac{\rho}{2}(\boldsymbol t_i - \boldsymbol s_i)^T(\boldsymbol t_i - \boldsymbol s_i)\\
=& \frac{1}{\rho^2}{\boldsymbol \mu_i}^T{\boldsymbol \mu_i} + \frac{\rho}{2}\frac{2}{\rho}{\boldsymbol \mu_i}^T({\boldsymbol{t}_i} - {\boldsymbol{s}_i}) + \frac{\rho}{2}(\boldsymbol t_i - \boldsymbol s_i)^T(\boldsymbol t_i - \boldsymbol s_i)\\
=& \frac{\rho}{2}[(\frac{1}{\rho}\boldsymbol \mu_i)^T(\frac{1}{\rho}\boldsymbol \mu_i) + \frac{1}{\rho}\boldsymbol \mu_i^T(\boldsymbol t_i - \boldsymbol s_i) + \frac{1}{\rho}\boldsymbol \mu_i(\boldsymbol t_i - \boldsymbol s_i)^T + (\boldsymbol t_i - \boldsymbol s_i)^T(\boldsymbol t_i - \boldsymbol s_i)]\\
=& \frac{\rho}{2}( \frac{1}{\rho}\boldsymbol \mu_i + (\boldsymbol t_i - \boldsymbol s_i) )^T( \frac{1}{\rho}\boldsymbol \mu_i + (\boldsymbol t_i - \boldsymbol s_i) )\\
=& \frac{\rho}{2}\vert\vert \frac{1}{\rho}\boldsymbol \mu_i + (\boldsymbol t_i - \boldsymbol s_i) \vert\vert_2^2
\end{aligned}
$$

则原始损失函数变为

$$
\min_{\boldsymbol t_i} \;    \sum\limits_i \lambda \vert\vert{\boldsymbol{t}_i}|{|_1} +\frac{\rho}{2}\vert\vert \frac{1}{\rho}\boldsymbol \mu_i + (\boldsymbol t_i - \boldsymbol s_i) \vert\vert_2^2
$$


上式是一个标准的 1 范数带二次项的形式，采用迭代收缩算法（具体解法参考下述链接）

> L1-L2范数最小化问题-迭代收缩算法. https://www.cnblogs.com/yuningqiu/p/9936184.html
<!-- > L1范数与L2范数的区别. https://zhuanlan.zhihu.com/p/28023308 -->

其标准闭环形式的更新公式为

$$
\boldsymbol t_i ^{(k+1)}= h_{\lambda\rho^{-1}}(\boldsymbol s_i^{(k)}-\rho^{-1}\boldsymbol \mu^{(k)})
$$

其中，$h_{\lambda\rho^{-1}}$ 为压缩算子。

### 3.1.3. u 更新

提出与 $\boldsymbol u_i$ 无关项，则有优化目标为

$$
\begin{aligned}
\min_{\boldsymbol u_i} L(\boldsymbol{u}) &=  {\eta _1}\sum\limits_{i,j} {u}_{ij}\vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2 \\
& + {\eta _2}\sum\limits_{i}I(i \in \Omega )\vert\vert{\hat{\boldsymbol{u}}_i} - {\boldsymbol{u}_i}\vert\vert_2^2 \\
& + < {\theta _{i}},\boldsymbol{u}_i\boldsymbol{1} - 1 > \\
& + \frac{\rho}
{2}\vert\vert\boldsymbol{u}_i\boldsymbol{1} - 1\vert\vert_2^2
\end{aligned}
$$

对于第一项

$$
\begin{aligned}
{\eta _1}\sum\limits_{i,j} {u}_{ij}\vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2 &= \sum\limits_{i}\eta _1\sum\limits_{j} {u}_{ij}\vert\vert{\boldsymbol{s}_i^{(k)}} - {\boldsymbol{c}_j}\vert\vert_2^2
\end{aligned}
$$

令 $\boldsymbol T^{(k)} \in \mathbb R^C$，其每个分量为

$$
\boldsymbol T_j^{(k)} = \vert\vert{\boldsymbol{s}_i} - {\boldsymbol{c}_j}\vert\vert_2^2, \quad j=1,2,\cdots, C
$$

可将第一个分量写成矩阵形式

$$
\sum\limits_{j} {u}_{ij}\vert\vert{\boldsymbol{s}_i^{(k)}} - {\boldsymbol{c}_j}\vert\vert_2^2 = tr(\boldsymbol u^T_i \cdot \boldsymbol T^{(k)})
$$

整个优化目标改写为

$$
\begin{aligned}
\min_{\boldsymbol u_i} L(\boldsymbol{u}) = & \sum\limits_{i}\eta _1\cdot  tr(\boldsymbol u^T_i \cdot \boldsymbol T^{(k)})\\
& + {\eta _2}\sum\limits_{i}I(i \in \Omega )\vert\vert{\hat{\boldsymbol{u}}_i} - {\boldsymbol{u}_i}\vert\vert_2^2 \\
& + < {\theta _{i}},\boldsymbol{u}_i\boldsymbol{1} - 1 > \\
& + \frac{\rho}{2}\vert\vert\boldsymbol{u}_i\boldsymbol{1} - 1\vert\vert_2^2\\
= & \eta _1\cdot  tr(\boldsymbol u^T_i \cdot \boldsymbol T^{(k)})\\
& + {\eta _2}I(i \in \Omega )\vert\vert{\hat{\boldsymbol{u}}_i} - {\boldsymbol{u}_i}\vert\vert_2^2 \\
& +  {\theta _{i}}(\boldsymbol{u}_i\boldsymbol{1} - 1 ) \\
& + \frac{\rho}{2}\vert\vert\boldsymbol{u}_i\boldsymbol{1} - 1\vert\vert_2^2
\end{aligned}
$$

其对 $\boldsymbol u_i$ 的偏导数为

$$
\begin{aligned}
&\eta_1\cdot \boldsymbol T^{(k)}\quad (103)\\
+&2{\eta _2}I(i \in \Omega )(\boldsymbol{u}_i - \hat{\boldsymbol{u}}_i)\\
+& {\theta _{i}}\boldsymbol{1}^T\quad (70)\\
+& \rho (\boldsymbol{u}_i\boldsymbol{1} - 1) \boldsymbol 1^T\quad (70)
\end{aligned}
$$

令偏导等于 0，得到参数 $\boldsymbol u_i$ 的更新式为

$$
\boldsymbol u_i^{(k+1)} = (2{\eta _2}I(i \in \Omega )+\rho\boldsymbol{1}\boldsymbol{1}^T)^{-1}
(2{\eta _2}I\hat{\boldsymbol{u}}_i-\eta_1\cdot \boldsymbol T^{(k)}+(\rho_{i}^{(k)}-\theta_{i}^{(k)})\boldsymbol 1^T)
$$

### 3.1.4. D 更新

优化目标为

$$
\min \sum\limits_i  \vert\vert \boldsymbol{x}_i - \boldsymbol{D}{\boldsymbol{s}_i^{(k)}}\vert\vert_2^2
$$

这是个典型的最小二乘回归问题，对其二范数展开为

$$
\sum\limits_i(\boldsymbol{x}_i - \boldsymbol{D}\boldsymbol{s}_i^{(k)})^T(\boldsymbol{x}_i - \boldsymbol{D}\boldsymbol{s}_i^{(k)}) =\\
\sum\limits_i[\boldsymbol{x}_i^T\boldsymbol{x}_i - \boldsymbol x_i^T\boldsymbol{D}\boldsymbol{s}_i^{(k)} - \boldsymbol x_i(\boldsymbol{D}\boldsymbol{s}_i^{(k)})^T + (\boldsymbol{D}\boldsymbol{s}_i^{(k)})^T\boldsymbol{D}\boldsymbol{s}_i^{(k)}]
$$

其对 $\boldsymbol D$ 的偏导为（参考Matrix Cookbook 式（70）、式（71）和式（77）)

$$
\sum\limits_i[-\boldsymbol x_i(\boldsymbol{s}_i^{(k)})^T-\boldsymbol{s}_i(\boldsymbol x_i^{(k)})^T+(\boldsymbol{s}_i^{(k)})^T\boldsymbol{D}\boldsymbol{s}_i^{(k)}+\boldsymbol{D}\boldsymbol{s}_i^{(k)}(\boldsymbol{s}_i^{(k)})^T]\\
=\sum\limits_i[-2\boldsymbol x_i(\boldsymbol{s}_i^{(k)})^T+2\boldsymbol{D}\boldsymbol{s}_i^{(k)}(\boldsymbol{s}_i^{(k)})^T]
$$

> 对于向量乘法可交换顺序，即$a^T\cdot b = a\cdot b^T$

令偏导等于 0，则更新式为

$$
\boldsymbol D^{(k+1)} = (\sum\limits_i\boldsymbol{s}_i^{(k)}(\boldsymbol{s}_i^{(k)})^T)^{-1}(\sum\limits_i\boldsymbol{x}_i^{(k)}(\boldsymbol{s}_i^{(k)})^T)
$$
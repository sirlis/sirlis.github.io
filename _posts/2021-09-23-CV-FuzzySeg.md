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
- [2. 模糊乘法稀疏编码](#2-模糊乘法稀疏编码)
  

# 1. 参数定义

- 扩散张量 $\boldsymbol{T}$：

$$
\boldsymbol{T}=[\boldsymbol v_1,\boldsymbol v_2,\boldsymbol v_3]
\begin{bmatrix}
e_1 & 0 &0 \\
0 & e_2 &0\\
0 & 0 & e_3
\end{bmatrix}
[\boldsymbol v_1,\boldsymbol v_2,\boldsymbol v_3]^T
$$

其中

- $e_1\geq e_2 \geq e_3$ 是特征值；
- $\boldsymbol{v}_1, \boldsymbol{v}_2, \boldsymbol{v}_3$ 是特征向量；
- $\boldsymbol{v}_1=[v_{11}, v_{12}, v_{13}]$ 是主要扩散方向，即最大特征值 $e_1$ 对应的特征向量；

描述符：

- 几何描述符 $\boldsymbol{x}_i^G$：

$$
\boldsymbol{x}_i^G=[e_1, e_2,e_3,FA, MD, RA, VR, RD, PA, SA, LA]\in \mathbb R^{11}
$$

- 方向描述符 $\boldsymbol{x}_i^O$：

$$
\boldsymbol{x}_i^O=\frac{1}{\vert\vert\boldsymbol{v}_1\vert\vert} (v_{11}^2-v_{12}^2, 2v_{11}v_{12}, 2v_{11}v_{13}, 2v_{12}v_{13},\frac{1}{\sqrt{3}}(2v_{13}^2-v_{11}^2-v_{12}^2)) \in \mathbb R^{5}
$$

- 原始数据域下，核磁共振体素的特征向量 $\boldsymbol{x}_i$：

$$
\boldsymbol{x}_i = [t_1, \boldsymbol x_i^G,\boldsymbol x_i^O]\in \mathbb R^k, \; k=17
$$

- 稀疏编码域下，体素特征向量的稀疏表示 $\boldsymbol s_i\in \mathbb R^l$。

- 超完备词典 $\boldsymbol D = [\boldsymbol d_1, \boldsymbol d_2, \cdots, \boldsymbol d_l]\in \mathbb R^{k\times l}$

# 2. 模糊乘法稀疏编码

输入训练样本 $\boldsymbol X = [\boldsymbol x_1, \cdots, \boldsymbol x_n]\in \mathbb R^{k\times n}$，对于每个样本可以训练字典得到稀疏编码

$$
F(\boldsymbol s, \boldsymbol D)={min_{\boldsymbol s, \boldsymbol D} \sum_{i=1}^n\frac{1}{2}\vert\vert\boldsymbol x_i - \boldsymbol D \boldsymbol s_i\vert\vert_2^2+\lambda\vert\vert \boldsymbol s_i\vert\vert_1,\quad s.t.\vert\vert\boldsymbol d_j\vert\vert_2^2 < 1 \; \forall j}
$$
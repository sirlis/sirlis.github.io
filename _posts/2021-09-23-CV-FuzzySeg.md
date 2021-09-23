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

 - [1. 网络结构](#1-网络结构)
  

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

- 核磁共振体素的特征向量 $\boldsymbol{x}_i$：

$$
\boldsymbol{x}_i = [t_1, \boldsymbol x_i^G,\boldsymbol x_i^O]\in \mathbb R^k, \; k=17
$$

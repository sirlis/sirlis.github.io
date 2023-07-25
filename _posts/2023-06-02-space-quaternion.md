---
title: 航天中的四元数以及姿态运动学
date: 2023-06-02 18:38:19 +0800
categories: [Knowledge]
tags: [space, quaternion, attitude kinematics]
math: true
---

本文介绍了航天器姿态描述、姿态变换和姿态运动学中涉及的四元数表示法。

<!--more-->

---

- [姿态四元数](#姿态四元数)
- [向量的坐标变换](#向量的坐标变换)
- [姿态变换与四元数乘法](#姿态变换与四元数乘法)
- [参考文献](#参考文献)


 
## 姿态四元数

在航天器相对位姿跟踪任务中，需要明确各个坐标系，并以此来描述刚体的位姿和受力情况。通常会涉及到以下几个相关坐标系。
- 地心惯性坐标系 $I$；
- 追踪航天器本体坐标系 $B$；
- 目标航天器本体坐标系 $T$； 
- 期望坐标系 $D$。

当四元数的范数限制为 1 时，就叫**单位四元数**，即

$$
\mathbb{H}^u=\{q\in \mathbb{H}: q\cdot q=(1,0,0,0)\}
$$

其可以用来表示两个坐标系之间的相对姿态，此时又可称其为**旋转四元数**。若本体坐标系 $B$ 相对于惯性系 $I$ 的姿态用欧拉旋转轴角表示为 $(\bar{n},\theta)$，即本体坐标系 $B$ 绕着单位轴 $\bar{n}$ 旋转角度 $\theta$ 到达惯性坐标系 $I$。那么本体坐标系相对于惯性坐标系的姿态四元数为

$$
q_{B/I}=(cos(\frac{\theta}{2}),\ sin(\frac{\theta}{2}\bar{n}))
$$

如果旋转角度被限制在 $-180^{\circ}< \theta < 180^{\circ}$，那么四元数标量部分可以由下式计算

$$
q_0 = \sqrt(1-\vert\vert \bar{q} \vert\vert ^2)
$$

注意，相对姿态轴角表示中， $(-\bar{n},-\theta)$ 和 $(\bar{n},\theta)$ 起到的旋转效果相同。带入姿态四元数定义式，可以得到惯性系相对于本体系的姿态四元数

$$
q_{I/B} = q_{B/I}^*
$$

即姿态四元数（单位四元数）本身与其共轭四元数互为逆变换。

## 向量的坐标变换

【**定理1**】假设坐标系 $Y$ 相对于坐标系 $X$ 的姿态四元数为 $q_{Y/X}$，矢量 $\bar{v}$ 再两个坐标系内的表示分别为 $\bar{v}^Y$ 和 $\bar{v}^X$，则有如下转换关系成立

$$
v^Y = q^*_{Y/X}\cdot v^X \cdot q_{Y/X},\quad v^X = q^*_{X/Y}\cdot v^Y \cdot q_{X/Y}
$$

其中 $v^Y = [0,\bar{v}^Y],\ v^X = [0,\bar{v}^X]$

## 姿态变换与四元数乘法

假设坐标系 $A$ 相对于坐标系 $B$ 的四元数为 $q_{A/B}$，坐标系 $B$ 相对于坐标系 $C$ 的四元数为 $q_{B/C}$，那么坐标系 $A$ 相对于坐标系 $C$ 的四元数为

$$
q_{A/C} = q_{A/B} \otimes q_{B/C}
$$

## 参考文献

[1] 皮皮夏. [【知乎】四元数代数以及姿态动力学建模](https://zhuanlan.zhihu.com/p/375199378)
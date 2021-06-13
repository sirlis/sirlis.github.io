---
title: 计算机视觉（One-Stage 目标检测）
date: 2021-06-08 09:08:49 +0800
categories: [Academic, Knowledge]
tags: [deeplearning]
math: true
---

本文介绍了计算机视觉中单阶段目标检测问题的解决方法，即 Yolo 系列。

<!--more-->

 ---
 
- [1. 前言](#1-前言)
- [2. YOLO V1](#2-yolo-v1)
- [6. 参考文献](#6-参考文献)

# 1. 前言

Yolo，SSD 这类 one-stage 算法，仅仅使用一个卷积神经网络 CNN 直接预测不同目标的类别与位置。一阶段方法的速度快，但是准确性要低一些。

# 2. YOLO V1

YOLO意思是You Only Look Once，创造性的将候选区和对象识别这两个阶段合二为一，看一眼图片（不用看两眼哦）就能知道有哪些对象以及它们的位置。

实际上，YOLO并没有真正去掉候选区，而是采用了预定义的候选区（准确点说应该是预测区，因为并不是Faster RCNN所采用的Anchor）。也就是将图片划分为 $7\times 7=49$ 个网格（grid），每个网格允许预测出 2 个边框（bounding box，包含某个对象的矩形框），总共 $49\times 2=98$ 个 bounding box。可以理解为 98 个候选区，它们很粗略的覆盖了图片的整个区域。

# 6. 参考文献

[1] 维基百科. [Kernel regression](https://en.wikipedia.org/wiki/Kernel_regression)

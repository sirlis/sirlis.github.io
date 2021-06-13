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
  - [2.1. 输入](#21-输入)
  - [2.2. 输出](#22-输出)
  - [2.3. 构造训练样本](#23-构造训练样本)
  - [损失函数](#损失函数)
  - [2.4. 测试](#24-测试)
- [3. 参考文献](#3-参考文献)

# 1. 前言

Yolo，SSD 这类 one-stage 算法，仅仅使用一个卷积神经网络 CNN 直接预测不同目标的类别与位置。一阶段方法的速度快，但是准确性要低一些。

# 2. YOLO V1

YOLO意思是You Only Look Once，创造性的将候选区和对象识别这两个阶段合二为一，看一眼图片（不用看两眼哦）就能知道有哪些对象以及它们的位置。

实际上，YOLO并没有真正去掉候选区，而是采用了预定义的候选区（准确点说应该是预测区，因为并不是Faster RCNN所采用的Anchor）。也就是将图片划分为 $7\times 7=49$ 个网格（grid），每个网格允许预测出 2 个边框（bounding box，包含某个对象的矩形框），总共 $49\times 2=98$ 个 bounding box。可以理解为 98 个候选区，它们很粗略的覆盖了图片的整个区域。

## 2.1. 输入

输入就是原始图像，唯一的要求是缩放到 $448\times 448$ 的大小。主要是因为YOLO的网络中，卷积层最后接了两个全连接层，全连接层是要求固定大小的向量作为输入，所以倒推回去也就要求原始图像有固定的尺寸。

## 2.2. 输出

输出是一个 $7\times 7\times 30$ 的张量。$7\times 7$ 对应原始图像的网格，30维向量 = 20个对象的概率 + 2个bounding box * 4个坐标 + 2个bounding box的置信度。

![](../assets/img/postsimg/20210613/yolov1output.jpg)

**前 20 维**，one hot 编码。因为YOLO支持识别20种不同的对象（人、鸟、猫、汽车、椅子等），所以这里有20个值表示该网格位置存在任一种对象的概率。

**中 2 维**，2 个 bounding box 的置信度。 = 该 bounding box 内有对象的概率 * 该 bounding box 与该对象实际 bounding box 的 IOU。

**后 8 维**，2 个 bounding box 的位置。每个 bounding box 需要 4 个数值来表示其位置，(Center_x, Center_y, width, height)，2 个 bounding box 共需要 8 个数值来表示其位置。

$7\times 7$网格，每个网格2个bounding box，对 $448\times 448$ 输入图像来说覆盖粒度有点粗。我们也可以设置更多的网格以及更多的bounding box。设网格数量为 $S\times S$，每个网格产生 B 个边框（4 位置 + 1 置信度），网络支持识别 C 个不同的对象。这时，输出的向量长度为： $C + B\times (4+1)$ 整个输出的tensor就是： $S\times S\times (C + B\times (4+1))$。

## 2.3. 构造训练样本

- **20 个对象分类的概率**

对于输入图像中的每个对象，先找到其中心点。中心点落在某个网格内，该网格对应 30 维向量中的 1 维置 1，其它维度置 0。所有其它 48 个网格的30维向量中，该对象的概率都是 0（也即一个网格只能预测 1 个对象，网络一共能从一张图片中检测49个对象）。这就是所谓的"中心点所在的网格对预测该对象负责"。

- **2 个 bounding box 的位置**

训练样本的 bounding box 位置应该填写对象实际的bounding box，但一个对象对应了 2 个 bounding box，该填哪一个呢？上面讨论过，需要根据网络输出的bounding box 与对象实际 bounding box 的 IOU 来选择，所以要在训练过程中动态决定到底填哪一个 bounding box。参考下面第 3 点。

- **2 个 bounding box 的置信度**

上面讨论过置信度公式

$$
Confidence = Pr(Object) * IOU^{truth}_{pre}
$$

2 个 bounding box 的 $IOU$，哪个比较大就由哪个 bounding box 负责预测该对象是否存在，相应的 $P(Object)=1$，$Confidence = IOU$，该网格其它 bounding box 的 $Confidence = 0$。

![](../assets/img/postsimg/20210613/trainlabel.jpg)

注意，在训练过程中等网络输出以后，比较两个 bounding box 与自行车实际位置的 IOU，自行车的位置（实际 bounding box）放置在 IOU 比较大的那个 bounding box（图中假设是 bounding box1），且该 bounding box 的置信度设为 1。

## 损失函数

![](../assets/img/postsimg/20210613/loss.jpg)



## 2.4. 测试

测试时，每个网格预测的 class 信息和 bounding box 预测的 confidence 信息相乘，就得到每个 bounding box 的 class-specific confidence score：

$$
Pr(C_i) = Pr(C_i\vert Object) *Pr(Object) * IOU_{pred}^{truth}
$$

得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的 boxes，对保留的 boxes 进行 NMS 处理，就得到最终的检测结果。

# 3. 参考文献


[1] X猪. [YOLO v1深入理解](https://www.jianshu.com/p/cad68ca85e27). 简书

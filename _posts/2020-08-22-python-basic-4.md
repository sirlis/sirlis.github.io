---
layout: post
title:  "Python基础（绘图plot）"
date:   2020-08-22 15:58:19
categories: Coding
tags: Python
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
* [mat](#mat)
  * [MATLAB保存mat文件](#MATLAB保存mat文件)
  * [Python读取mat文件](#Python读取mat文件)
* [list](#list)
  * [复制](#复制)
  * [合并](#合并)
  * [插入新元素](#插入新元素)
  * [获取列表中的值](#获取列表中的值)
* [ndarray](#ndarray)
  * [概念](#概念)
  * [数组属性](#数组属性)
  * [创建数组](#创建数组)
* [参考文献](#参考文献)

# 二维图

## plot

`plot` 用以展示变量的趋势变化。`plot()` 函数的本质就是根据点连接线。根据x(数组或者列表) 和y(数组或者列表)组成点，然后连接成线。

简单示例如下

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.05, 10, 1000)
y = np.cos(x)

plt.plot(x, y, ls="-", lw=2, label="plot figure")
plt.legend()
plt.show()
```

![image-20200822155756531](..\assets\img\postsimg\20200822\1.jpg)

## 颜色控制

要想使用丰富，炫酷的图标，我们可以使用更复杂的格式设置，主要颜色，线的样式，点的样式。默认的情况下，只有一条线，是蓝色实线。多条线的情况下，生成不同颜色的实线。

| 字符 | 颜色        |
| ---- | ----------- |
| 'b'  | blue        |
| 'g'  | green       |
| 'r'  | red         |
| 'c'  | cyan 青色   |
| 'm'  | magenta平红 |
| 'y'  | yellow      |
| 'k'  | black       |
| 'w'  | white       |

## 线形控制

| 字符 | 类型             |
| ---- | ---------------- |
| '-'  | 实线             |
| '--' | 虚线             |
| '-.' | 虚点线           |
| ':'  | 点线             |
| ' '  | 空类型，不显示线 |

例如

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y1 = [1, 2, 3, 4]
y2 = [1, 4, 9, 16]
y3 = [1, 8, 27, 64]
y4 = [1, 16, 81, 124]
# 创建一个画布
plt.figure()
# 在figure下线
plt.plot(x, y1, "-o") #实线
plt.plot(x, y2, "--o") #虚线
plt.plot(x, y3, "-.o") #虚点线
plt.plot(x, y4, ":o") # 点线
# 展现画布
plt.show()
```

绘制效果为

![image-20200822160154771](..\assets\img\postsimg\20200822\2.jpg)

## 点型控制

| 点型 | 类型       |
| ---- | ---------- |
| '.'  | 点         |
|      |            |
| ','  | 像素点     |
| 'o'  | 原点       |
| '^'  | 上三角点   |
| 'v'  | 下三角点   |
| '<'  | 左三角点   |
| '>'  | 右三角点   |
| '1'  | 下三叉点   |
| '2'  | 上三叉点   |
| '3'  | 左三叉点   |
| '4'  | 右三叉点   |
| 's'  | 正方点     |
| '+'  | 加号点     |
| 'x'  | 乘号点     |
| 'D'  | 实心菱形点 |
| 'd'  | 细菱形点   |

示例

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y1 = [1, 2, 3, 4]
y2 = [1, 4, 9, 16]
y3 = [1, 8, 27, 64]
y4 = [1, 16, 81, 124]
# 创建一个画布
plt.figure()
# 在figure下的线
plt.plot(x, y1, "-.") # 点
plt.plot(x, y2, "-,") # 像素点
plt.plot(x, y3, "-o") # 圆点

# 展现画布
plt.show()
```

绘制效果为

![image-20200822160526163](..\assets\img\postsimg\20200822\3.jpg)

## 另一种设置方法

color="green" 指定颜色为绿色

linestyle="dashed" 指定线形为dashed类型

marker="o" 指定标记类型为o点

markerfacecolor="blue"指定标记的颜色为蓝色

markersize=20 指定标记的大小为20

比如

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y1 = x * 1.5
y2 = x * 2.5
y3 = x * 3.5
y4 = x * 4.5
y5 = x * 5.5

plt.plot(x, y1, "-P")
plt.plot(x, y2, "-|")
plt.plot(x, y3, color="#000000")
plt.plot(x, y4, "-o", markersize=20)
plt.plot(x, y5, "-^", markerfacecolor="blue")

plt.show()
```

![image-20200822162131136](..\assets\img\postsimg\20200822\4.jpg)

# subplot

有些时候, 我们希望把一组图放在一起进行比较, 有没有什么好的方法呢？`matplotlib` 中提供的 `subplot` 可以很好的解决这个问题。`matplotlib` 下, 一个 `Figure` 对象可以包含多个子图(`Axes`), 可以使用 `subplot()` 快速绘制, 其调用形式如下 :

```python
subplot(numRows, numCols, plotNum)
```

- 图表的整个绘图区域被分成 `numRows` 行和 `numCols` 列；
- 然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1；
- `plotNum` 参数指定创建的 `Axes` 对象所在的区域；
- 如果 `numRows`, `numCols` 和 `plotNum` 这三个数都小于 `10` 的话, 可以把它们缩写为一个整数, 例如 `subplot(323)` 和 `subplot(3,2,3)` 是相同的；
- `subplot`在 `plotNum` 指定的区域中创建一个轴对象. 如果新创建的轴和之前创建的轴重叠的话，之前的轴将被删除。



# 参考文献

[1] [梦并不遥远](https://www.cnblogs.com/zyg123/). [4.3Python数据处理篇之Matplotlib系列(三)---plt.plot()](https://www.cnblogs.com/zyg123/p/10504633.html).

[2] RUNOOB.COM. [NumPy 从数值范围创建数组](https://www.runoob.com/numpy/numpy-array-from-numerical-ranges.html).
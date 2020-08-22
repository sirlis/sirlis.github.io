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

| 点型 | 类型     |
| ---- | -------- |
| '.'  | 点       |
| ','  | 像素点   |
| 'o'  | 原点     |
| '^'  | 上三角点 |
| 'v'  | 下三角点 |
| '<'  | 左三角点 |
| '>'  | 右三角点 |
| '1'  | 下三叉点 |
| '2'  | 上三叉点 |
| '3'  | 左三叉点 |
| '4'  | 右三叉点 |

's'正方点'p'五角点'*'星形点'h'六边形1'H'六边形2

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

# list

列表（list）是用来存储一组有序数据元素的数据结构，元素之间用都好分隔。列表中的数据元素应该包括在**方括号**中，而且列表是可变的数据类型，一旦创建了一个列表，你可以添加、删除或者搜索列表中的元素。在方括号中的数据可以是 `int` 型，也可以是 `str` 型。

新建一个空列表

```python
A = []
```

当方括号中的数据元素全部为int类型时，这个列表就是int类型的列表。str类型和混合类型的列表类似

```python
A = [1,2,3]
A = ["a",'b','c'] # 单引号和双引号都认
A = [1,"b",3]
```

## 复制

**列表的复制**和字符串的复制类似，也是利用 `*` 操作符

```python
A = [1,2,3]
A*2 # A = [1,2,3,1,2,3]
```

## 合并

**列表的合并**就是将两个现有的list合并在一起，主要有两种实现方式，一种是利用+操作符，它和字符串的连接一致；另外一种用的是extend()函数。

直接将两个列表用+操作符连接即可达到合并的目的，列表的合并是有先后顺序的。

```python
a = [1,2]
b = ['a','b']
m = ["c","c"]
c=a+b+m # c = [1,2,'a','b','c','c']
d=b+a+m # d = ['a','b',1,2,'c','c']
```

将列表b合并到列表a中，用到的方法是a.extend(b)，将列表a合并到列表b中，用到的方法是b.extend(a)。

## 插入新元素

向列表中**插入新元素**。列表是可变的，也就是当新建一个列表后你还可以对这个列表进行操作，对列表进行插入数据元素的操作主要有 `append()` 和 `insert()` 两个函数可用。这两个函数都会直接改变原列表，不会直接输出结果，需要调用原列表的列表名来获取插入新元素以后的列表。

函数 `append()` 是在列表末尾插入新的数据元素，如下：

```python
a = [1,2]
a.append(3) # a = [1,2,3]
```

函数 `insert()` 是在列表指定位置插入新的数据元素，如下：

```python
a = [1,2,3]
a.insert(3,4) # a = [1,2,3,4]，在列表第四位（从0开始算起）插入4
a = [1,2]
a.insert(2,4) # a = [1,2,4,3]，在列表第三位（从0开始算起）插入4
```

## 获取列表中的值

获取指定位置的值利用的方法和字符串索引是一致的，主要是有普通索引和切片索引两种。

- 普通索引：普通索引是活期某一特定位置的数，如下：

```python
>>> a = [1,2,3]
>>> a[0] # 获取第一位数据
1
>>> a[2]
3
```

- 切片索引：切片索引是获取某一位置区间内的数，如下：

```python
>>> a = [1,2,3,4,5]
>>> a[1:3] # 获取第2位到第4位的数据，不包含第4位
[2,3]
```

假设 `a = [1,2,3,4,5,6,7,8,9]`，对应的标号为 `[0,2,2,3,4,5,6,7,8]`；

`print a[1:2:3]` 输出为2 ，从下标表为1的地方开始到小于下标为2的位置，其中3为步长；

`print a[1:4:1]` 输出为2，3，4,以上面类似，只是步长为1了；

`print a[1::1]` 输出为2，3，4，5，6，7，8，9。中间为空表示默认，则从小标为1到最后；

`print a[-1:-4:-1]` 反向索引，从最后一位开始放过来取值，注意这里的步长要为-1，因为反向。

# ndarray

Numpy是Python的一个扩充程序库，支持高阶大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。对于数据的运算，用矩阵会比python自带的字典或者列表快好多。

## 概念

NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列**同类型**数据的集合，以 0 下标为开始进行集合中元素的索引。

ndarray 对象是用于存放同类型元素的多维数组。

ndarray 中的每个元素在内存中都有相同存储大小的区域。

ndarray 内部由以下内容组成：

- 一个指向数据（内存或内存映射文件中的一块数据）的指针。
- 数据类型或 dtype，描述在数组中的固定大小值的格子。
- 一个表示数组形状（shape）的元组，表示各维度大小的元组。
- 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。

创建一个 ndarray 只需调用 NumPy 的 array 函数即可：

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```

参数说明

| 名称     | 描述                                                      |
| :------- | :-------------------------------------------------------- |
| `object` | 数组或嵌套的数列                                          |
| `dtype`  | 数组元素的数据类型，可选                                  |
| `copy`   | 对象是否需要复制，可选                                    |
| `order`  | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| `subok`  | 默认返回一个与基类类型一致的数组                          |
| `ndmin`  | 指定生成数组的最小维度                                    |

例子

```python
>>> import numpy as np 
>>> a = np.array([1,2,3])  
>>> print (a)
[1, 2, 3]
>>> a = np.array([[1,  2],  [3,  4]])  
>>> print (a)
[[1, 2] 
 [3, 4]]
```

## 数组属性

NumPy 数组的维数称为秩（rank），秩就是轴的数量，即数组的维度，一维数组的秩为 1，二维数组的秩为 2，以此类推。

在 NumPy中，每一个线性的数组称为是一个轴（axis），也就是维度（dimensions）。比如说，二维数组相当于是两个一维数组，其中第一个一维数组中每个元素又是一个一维数组。所以一维数组就是 NumPy 中的轴（axis），第一个轴相当于是底层数组，第二个轴是底层数组里的数组。而轴的数量——秩，就是数组的维数。

很多时候可以声明 axis。axis=0，表示沿着第 0 轴进行操作（沿着行移动），即对每一列进行操作；axis=1，表示沿着第1轴进行操作（沿着列移动），即对每一行进行操作。

## 创建数组

- `numpy.zeros` 创建指定大小数组，数组元素以 0 来填充：

```python
>>> import numpy as np 
>>> x = np.zeros(5) 
>>> print (x)
[0. 0. 0. 0. 0.]
>>> y = np.zeros((5,), dtype = np.int) 
>>> print (y)
[0 0 0 0 0]
```

- `numpy.ones` 创建指定形状的数组，数组元素以 1 来填充：

```python
>>> import numpy as np 
[0. 0. 0. 0. 0.]
>>> x = np.ones(5)
[1. 1. 1. 1. 1.]
>>> print (x)
>>> y = np.ones((2,2), dtype = np.int) 
>>> print (y)
[[1 1]
 [1 1]]
```

- `numpy.arange` 创建数值范围并返回 ndarray 对象，函数格式如下：

```python
numpy.arange(start, stop, step, dtype)
```

生成数组示例如下：

```python
>>> import numpy as np
>>> x = np.arange(5) # = np.arrange(0,1,5)
>>> print (x)
[0  1  2  3  4]
>>> x = np.arange(5, dtype =  float)
>>> print (x)
[0.  1.  2.  3.  4.]
>>> x = np.arange(10,20,2)  
>>> print (x)
[10  12  14  16  18]
>>> x = np.arange(0,1,0.1)
>>> print (x)
[ 0.  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9]
```

- `numpy.reshape` 在不改变数据内容的情况下，改变一个数组的格式，参数及返回值：

```python
>>> import numpy as np
>>> x = np.arange(6)
>>> print (x)
[0  1  2  3  4  5]
>>> y = x.reshape((2,3))
>>> print (y)
[[0 1 2]
 [3 4 5]]
>>> z = x.reshape(-1,2)
>>> print (z)
[[0 1]
 [2 3]
 [4 5]]
```

通过 reshape 生成的新数组和原始数组共用一个内存，也就是改变了原数组的元素，新数组的相应元素也将发生改变。

-1 表示要根据另一个维度自动计算当前维度。`reshape(-1,2)` 即我们想要2列而不知道行数有多少，让numpy自动计算。

- `numpy.linspace` 创建一个一维数组，数组是一个等差数列构成的，格式如下：

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

参数说明：

| 参数       | 描述                                                         |
| :--------- | :----------------------------------------------------------- |
| `start`    | 序列的起始值                                                 |
| `stop`     | 序列的终止值，如果 `endpoint` 为 `true`，该值包含于数列中    |
| `num`      | 要生成的等步长的样本数量，默认为 `50`                        |
| `endpoint` | 该值为 `true` 时，数列中包含 `stop` 值，反之不包含，默认是 `true`。 |
| `retstep`  | 如果为 `true` 时，生成的数组中会显示间距，反之不显示。       |
| `dtype`    | `ndarray` 的数据类型                                         |

示例：

```python
>>> import numpy as np
>>> a = np.linspace(1,10,10)
>>> print(a)
[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
```

## 数组拼接

两个拼接数组的方法：

`np.vstack()` 在竖直方向上堆叠

`np.hstack()` 在水平方向上平铺

```python
>>> import numpy as np
>>> arr1=np.array([1,2,3])
>>> arr2=np.array([4,5,6])
>>> print np.vstack((arr1,arr2))
[[1 2 3]
 [4 5 6]]
>>> print np.hstack((arr1,arr2,arr1))
[1 2 3 4 5 6 1 2 3]
>>> a1=np.array([[1,2],[3,4],[5,6]])
>>> a2=np.array([[7,8],[9,10],[11,12]])
>>> print np.hstack((a1,a2))
[[ 1  2  7  8]
 [ 3  4  9 10]
 [ 5  6 11 12]]
```

# list和array的区别

在数据类型上，二者存在区别

```python
>>> import numpy as np
>>> a=np.array([1,2,3,4,55,6,7,77,8,9,99]) # array
>>> b=np.array_split(a,3) # 分为三段
>>> print (b) # list, 包含3个 array 元素
[array([1, 2, 3, 4]), array([55,  6,  7, 77]), array([ 8,  9, 99])]
>>> print (b[0:2]+b[1:3]) # list，包含4个 array 元素
[array([1, 2, 3, 4]), array([55,  6,  7, 77]), array([55,  6,  7, 77]), array([ 8,  9, 99])]
>>> c = np.hstack((brr1_folds[:2]+brr1_folds[1:3]))
>>> print (c) # list
[ 1  2  3  4 55  6  7 77 55  6  7 77  8  9 99]
```

# 参考文献

[1] CDA数据分析师. [Python基础知识详解(三)：数据结构篇](https://baijiahao.baidu.com/s?id=1652154442455041874&wfr=spider&for=pc).

[2] RUNOOB.COM. [NumPy 从数值范围创建数组](https://www.runoob.com/numpy/numpy-array-from-numerical-ranges.html).
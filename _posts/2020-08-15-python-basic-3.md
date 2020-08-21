---
layout: post
title:  "Python基础（数据结构）"
date:   2020-08-15 21:17:19
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
* [np.random](#np.random)
  * [.seed()](#.seed())
  * [.RandomState()](#.RandomState())0
  * [.choice()](#.choice())
  * [.uniform()](#.uniform())
  * [.permutation()](#.permutation())
* [参考文献](#参考文献)

# mat

## MATLAB保存mat文件

如下例子所示

```matlab
clear
clc

maindir = 'Results';
subdir  = dir( maindir );

j = 1;
for ii = 1 : length( subdir )
    if( isequal( subdir( ii ).name, '.' )||...
        isequal( subdir( ii ).name, '..')||...
        ~subdir( ii ).isdir) % skip if not dir
        continue;
    end
    
    matfile = fullfile( maindir, subdir( ii ).name, 'Result.mat' );
    condition = split(subdir( ii ).name, '_');
    load(matfile)
    dataCell{j,1} = condition(2);
    dataCell{j,2} = condition(3);
    dataCell{j,3} = DesireStatus;
    dataCell{j,4} = DesireControl;
    j = j + 1;
end

save('MixResults.mat','dataCell');
```

最终保存的文件形式如图所示

![image-20200814214150568](..\assets\img\postsimg\20200815\1.jpg)

## Python读取mat文件

```python
import scipy.io as sio
import numpy as np

load_path = 'MixResults.mat'
load_data = sio.loadmat(load_path)
```

此处得到的 `load_data` 是如下形式的值

```python
{'__globals__': [],
'__header__': b'MATLAB 5.0 MAT-file,...00:47 2020',
'__version__': '1.0',
'dataCell': array([[array([[arra...pe=object)}
```

其中，scipy读入的mat文件是个dict类型，会有很多不相关的keys，一般输出四种keys：\_\_globals\_\_，\_\_header\_\_，\_\_version\_\_，data。其中最后一个data才是我们要的数据。

本例中，数据为 `dataCell` ，内容为

```
array([[array([[array(['wty0.0'], dtype='<U6')]], dtype=object),
        array([[array(['-30deg'], dtype='<U6')]], dtype=object),
        array([[ 0.00000000e+00, -8.66025400e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 1.31815433e-01, -8.66003375e+00,  3.20285361e-12, ...,
        -6.21551560e-19,  0.00000000e+00,  0.00000000e+00],
       [ 2.63630865e-01, -8.65933419e+00,  1.29989166e-11, ...,
        -3.17675810e-19,  0.00000000e+00,  0.00000000e+00],
       ...,
       [ 4.58717705e+01, -2.75108745e+00,  2.14124086e-11, ...,
         2.86948290e-17, -5.53482295e-17,  5.27271879e-15],
       [ 4.60035859e+01, -2.75027966e+00, -4.46901743e-12, ...,
         1.32056047e-17, -4.37313113e-17, -2.42826591e-16],
       [ 4.61354014e+01, -2.75000000e+00,  2.89500769e-27, ...,
        -2.77555756e-17, -1.30104261e-17, -1.48286602e-14]]),
        array([[ 0.00000000e+00,  2.40730441e+00,  6.83370983e-08, ...,
         2.87639504e+00, -3.90482776e-07,  2.65632451e-01],
       [ 1.31815433e-01,  2.77247364e+00,  2.88899127e-04, ...,
         1.27018538e+00, -2.19925908e-07,  1.26745599e-01],
       [ 2.63630865e-01,  3.06865475e+00,  1.11303404e-03, ...,
         6.74119805e-02, -9.25865826e-08,  2.27535649e-02],
       ...,
       [ 4.58717705e+01, -3.25427728e+00,  5.09348887e-03, ...,
         2.01213062e-02,  1.28660157e-03,  2.75940695e+00],
       [ 4.60035859e+01, -3.15266928e+00,  1.21336612e-03, ...,
         1.97145055e-02,  1.30781251e-03,  3.95909773e+00],
       [ 4.61354014e+01, -3.04097330e+00,  1.01612056e-07, ...,
         1.87656241e-02,  1.31095528e-03,  5.18427291e+00]])],
       ...,
       dtype=object)
```

可以按照如下方式拼命取出 `dataCell` 中的各个元素

```python
>>> targetw = []
>>> targetw.append(load_data['dataCell'][i][0][0][0][0]) # wty
'wty0.0'
>>> position = []
>>> position.append(load_data['dataCell'][i][1][0][0][0]) # deg
'-30deg'
>>> trajectory = np.zeros((tnum,1))
>>> trajectory = load_data['dataCell'][i][2]
>>> control = np.zeros((tnum,1))
>>> control = load_data['dataCell'][i][3]
```

得到的数据为 ndarray ，数据结构为

![image-20200815232302271](..\assets\img\postsimg\20200815\2.jpg)

取出其中 `wty` 的数值，可以借助 `strip` 砍掉 “wty” 三个字符。注意 `strip` 后返回的是一个字符串，需要通过强制格式转换转为数字

```python
wt = wty.strip('wty') # 0.0
wt = float(wt)
```

提取 `trajectory` 中的第一行数据，最后一行数据，以及 `wt`，合并后形成一个新的向量，借助 `np.hstack` 实现：

```python
trow = trajectory.shape[0]
vec = np.hstack((trajectory[0,1:4],trajectory[trow-1,1:4],wt))
# trajectory: t, rx, ry, rz, vx, ...; [1:4] is rx ry and rz
```

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

## 数组中选取特定列

假设 `a = [1,2,3,4,5,6,7,8,9]`，对应的标号为 `[0,2,2,3,4,5,6,7,8]`；

`print a[1:2:3]` 输出为2 ，从下标表为1的地方开始到小于下标为2的位置，其中3为步长；

`print a[1:4:1]` 输出为2，3，4,以上面类似，只是步长为1了；

`print a[1::1]` 输出为2，3，4，5，6，7，8，9。中间为空表示默认，则从小标为1到最后；

`print a[-1:-4:-1]` 反向索引，从最后一位开始放过来取值，注意这里的步长要为-1，因为反向。

# 参考文献

[1] CDA数据分析师. [Python基础知识详解(三)：数据结构篇](https://baijiahao.baidu.com/s?id=1652154442455041874&wfr=spider&for=pc)

无。
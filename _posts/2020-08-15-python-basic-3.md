---
layout: post
title:  "Python基础（mat与cell）"
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
* [lambda](#lambda)
* [np.random](#np.random)
  * [.seed()](#.seed())
  * [.RandomState()](#.RandomState())0
  * [.choice()](#.choice())
  * [.uniform()](#.uniform())
  * [.permutation()](#.permutation())
* [参考文献](#参考文献)

# MATLAB保存mat文件

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

![image-20200814214150568](E:\GitHub\sirlis.github.io\assets\img\postsimg\20200815\1.jpg)

# Python读取mat文件

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
>>> condition = load_data['dataCell'][0][0][0][0][0]
'wty0.0'
>>> condition = load_data['dataCell'][0][1][0][0][0]
'-30deg'
```



## .seed()

`numpy.random.seed()` ：设置 `seed()` 里的数字就相当于设置了一个盛有随机数的“聚宝盆”，一个数字代表一个“聚宝盆”，当我们在 `seed()` 的括号里设置相同的seed，“聚宝盆”就是一样的，那当然每次拿出的随机数就会相同（不要觉得就是从里面随机取数字，只要设置的seed相同取出地随机数就一样）。如果不设置seed，则每次会生成不同的随机数。（注：seed括号里的数值基本可以随便设置哦）。

下面给个例子，表明每次rand之前都要设置一次seed，只要设置的seed相同，那么产生的随机数就相同。

```python
>>> np.random.seed(0)
>>> np.random.rand(4,3)
array([[0.5488135 , 0.71518937, 0.60276338],
       [0.54488318, 0.4236548 , 0.64589411],
       [0.43758721, 0.891773  , 0.96366276],
       [0.38344152, 0.79172504, 0.52889492]])
>>> np.random.seed(0)
>>> np.random.rand(4,3)
array([[0.5488135 , 0.71518937, 0.60276338],
       [0.54488318, 0.4236548 , 0.64589411],
       [0.43758721, 0.891773  , 0.96366276],
       [0.38344152, 0.79172504, 0.52889492]])
```

给随机生成器设置seed的目的是每次运行程序得到的随机数的值相同，这样方便测试。

但是 `numpy.random.seed()` 不是线程安全的，如果程序中有多个线程最好使用 `numpy.random.RandomState` 实例对象来创建或者使用`random.seed()` 来设置相同的随机数种子。

## .RandomState()

`numpy.random.RandomState()` 是一个伪随机数生成器。那么伪随机数是什么呢？

伪随机数是用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列。并不真正的随机，但具有类似于随机数的统计特征，如均匀性、独立性等。

下面我们来看看它的用法：

```python
>>> import numpy as np
>>> rng = np.random.RandomState(0)
>>> rng.rand(4)
array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
>>> rng = np.random.RandomState(0)
>>> rng.rand(4)
array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
```

看，是不是生成了一样的随机数组呢，这点和numpy.random.seed（）还是很一样的。

因为是伪随机数，所以必须在rng这个变量下使用，如果不这样做，那么就得不到相同的随机数组了，即便你再次输入了numpy.random.RandomState()：

```
np.random.RandomState(0)
Out[397]: <mtrand.RandomState at 0xddaa288>
np.random.rand(4)
Out[398]: array([0.62395295, 0.1156184 , 0.31728548, 0.41482621])
np.random.RandomState(0)
Out[399]: <mtrand.RandomState at 0xddaac38>
np.random.rand(4)
Out[400]: array([0.86630916, 0.25045537, 0.48303426, 0.98555979])
```

## .choice()

`choice()` 方法返回一个列表，元组或字符串的随机项。**注意：**choice()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。

```python
>>> import random
>>> print "choice([1, 2, 3, 5, 9]) : ", random.choice([1, 2, 3, 5, 9])
choice([1, 2, 3, 5, 9]) :  2
>>> print "choice('A String') : ", random.choice('A String')
choice('A String') :  n
```

给定size参数后，可以生成指定size的随机数，如果需要每一次产生的随机数相同，则需要设置随机数种子，`random.seed(int)` 或者 `random.RandomState(int)`。

```python
import numpy as np
seed = 0
rng = np.random.RandomState(seed)
rng.choice(50, 10)
```

`numpy.random.choice(a, size=None, replace=True, p=None)` 

- a : 如果是一维数组，就表示从这个一维数组中随机采样；如果是int型，就表示从0到a-1这个序列中随机采样。
- size : 采样结果的数量，默认为1.可以是整数，表示要采样的数量；也可以为tuple，如(m, n, k)，则要采样的数量为m * n * k，size为(m, n, k)。
- replace : boolean型，采样的样本是否要更换？这个地方我不太理解，测了一下发现replace指定为True时，采样的元素会有重复；当replace指定为False时，采样不会重复。
- p : 一个一维数组，制定了a中每个元素采样的概率，若为默认的None，则a中每个元素被采样的概率相同。

## .uniform()

**uniform()** 方法将随机生成下一个实数（浮点数），它在 **[x, y]** 范围内。**注意：**uniform()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。

```python
import random
random.uniform(x, y)
```

其中x和y是随机数的取值界限，且不包含本身。

## .permutation()

随机排列一个序列，返回一个排列的序列。

```python
>>> np.random.permutation(10)
array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
>>> np.random.permutation([1, 4, 9, 12, 15])
array([15,  1,  9,  4, 12])
```

与 **random.shuffle(x)** 的区别在于，shuffle直接改变原始数组x，而permutation不改变原数组，而是赋值给新数组。

如果x是int型，，则返回从0到x-1这个序列的随机顺序。

# range()

函数语法：range(start, stop[, step])

参数说明：

- `start`: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
- `stop`: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5；
- `step`：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)。

# 参考文献

无。
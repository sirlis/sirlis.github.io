---
layout: post
title:  "Python基础（lambda,np.random）"
date:   2020-08-12 19:06:19
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
* [super(XXX, self).\__init__()](#super(XXX, self).\__init__())
* [参考文献](#参考文献)

# lambda

除了 `def` 语句，python还提供了一种生成函数对象的表达式形式。由于它与LISP语言中的一个工具类似，所以称为lambda。

就像 `def` 一样，这个表达式创建了一个之后能够调用的函数，但是它返回一个函数而不是将这个函数赋值给一个变量。这些就是 `lambda` 叫做匿名函数的原因。实际上，他常常以一种行内进行函数定义的方式使用，或者用作推迟执行一些代码。

`lambda` 的一般形式是关键字 `lambda` 之后跟着一个或多个参数（与一个 `def` 头部内用括号括起来的参数列表类似），紧跟着是一个冒号，之后是表达式

```python
lambda arg1,arg2,argn: expression using arguments
```

由 `lambda` 表达式所返回的函数对象与由def创建并复制后的函数对象工作起来是完全一致的，但 `lambda` 有一些不同之处，让其扮演特定的角色时更有用：

**lambda是一个表达式，而不是一个语句**

因为这一点，`lambda` 可以出现在python语法不允许 `def` 出现的地方。此外，作为一个表达式，`lambda` 返回一个值（一个新的函数），可以选择性的赋值给一个变量。相反，`def` 语句总是得在头部将一个新的函数赋值给一个变量，而不是将这个函数作为结果返回。

**lambda的主题是单个表达式，而不是一个代码块**

这个 `lambda` 的主题简单的就好像放在 `def` 主体 `return` 语句中的代码一样。简单的将结果写成一个顺畅的表达式，而不是明确的返回。但由于它仅限于表达式，故 `lambda` 通常要比 `def` 功能少…你仅能够在 `lambda` 主体中封装有限的逻辑进去，因为他是一个为编写简单函数而设计的。除了上述这些差别，`def` 和 `lambda` 都能过做同样种类的工作。

**def与lambda的相同用法**

```python
x = lambda x, y, z: x + y + z
x(2, 3, 4)
>>> 9

y = (lambda a='hello', b='world': a + b)
y(b='Python')
>>> 'hellopython'
```

**为什么使用lambda**

通常来说，lambda起到一种函数的速写作用，允许在使用的代码内嵌一个函数的定义，他完全是可选的（是可以使用 `def` 代替他们），但是在你仅需要切入一段可执行代码的情况下，它会带来一个更简洁的书写效果。

`lambda` 通常用来编写跳转表，也就是行为的列表或者字典，能够按照需求执行操作，比如以下（**表示乘方）：

```python
l = [lambda x: x ** 2, lambda x: x ** 3]
for f in l:
    print(f(2))
>>> 4
>>> 8
print(l[0](3))
>>> 9
```

实际上，我们可以用python中的字典或者其他的数据结构来构建更多种类的行为表，从而做同样的事情。

**map 函数**

程序对列表或者其他序列常常要做的一件事就是对每个元素进行一个操作，并把其结果集合起来。

python提供了一个工具map，它会对一个序列对象中的每一个元素应用该的函数，并返回一个包含了所有函数调用结果的列表。

举个栗子，我们有一个列表，需要将列表的每一个字段+10，我们该如何操作？

```python
list_show = [1, 2, 3, 4]

# 方式1
new_list_show = []
for i in list_show:
    new_list_show.append(i + 10)
print(new_list_show)

# 方式2
def adds(x):
    return x + 10
print(list(map(adds, list_show)))

# 更优雅的方式3：
print(list(map(lambda x: x + 10, list_show)))
```

**filter函数**

`filter` 通过字面意思，大家就知道它的用处了，用于数据的过滤操作，它也是 `lambda` 的一个好基友，举个栗子。

我们需要过滤0-9中，能被2整除的数字组成一个列表，我们该如何操作？只需要一行代码：

```python
print(list(filter(lambda x: x % 2 == 0,range(10))))
>>> [0, 2, 4, 6, 8]
```

# np.random

`numpy.random.seed()` ：设置 `seed()` 里的数字就相当于设置了一个盛有随机数的“聚宝盆”，一个数字代表一个“聚宝盆”，当我们在 `seed()` 的括号里设置相同的seed，“聚宝盆”就是一样的，那当然每次拿出的随机数就会相同（不要觉得就是从里面随机取数字，只要设置的seed相同取出地随机数就一样）。如果不设置seed，则每次会生成不同的随机数。（注：seed括号里的数值基本可以随便设置哦）。

下面给个例子，表明每次rand之前都要设置一次seed，只要设置的seed相同，那么产生的随机数就相同。

```python
np.random.seed(0)
np.random.rand(4,3)
Out[362]: 
array([[0.5488135 , 0.71518937, 0.60276338],
       [0.54488318, 0.4236548 , 0.64589411],
       [0.43758721, 0.891773  , 0.96366276],
       [0.38344152, 0.79172504, 0.52889492]])
np.random.seed(0)
np.random.rand(4,3)
Out[364]: 
array([[0.5488135 , 0.71518937, 0.60276338],
       [0.54488318, 0.4236548 , 0.64589411],
       [0.43758721, 0.891773  , 0.96366276],
       [0.38344152, 0.79172504, 0.52889492]])
```

`numpy.random.RandomState()` 是一个伪随机数生成器。那么伪随机数是什么呢？

伪随机数是用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列。并不真正的随机，但具有类似于随机数的统计特征，如均匀性、独立性等。

下面我们来看看它的用法：

```python
import numpy as np

rng = np.random.RandomState(0)
rng.rand(4)
Out[377]: array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
rng = np.random.RandomState(0)
rng.rand(4)
Out[379]: array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
```



# 参考文献

无。
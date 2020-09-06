---
title: Python基础（万恶的下划线）
date: 2020-06-14 10:40:19 +0800
categories: [Coding]
tags: [python]
---

# \__name__ == '__main__'

`if __name__ == '__main__'` 为 python 提供模拟的应用程序入口。它的功能为：当.py文件被直接运行时，`if __name__ == '__main__':` 之下的代码块将被运行；当.py文件以模块形式被导入时，`if __name__ == '__main__':` 之下的代码块不被运行。

参考：https://blog.csdn.net/anshuai_aw1/article/details/82344884

假设我们有一个const.py文件，内容如下：

```python
PI = 3.14
 
def main():
    print("PI:", PI)
 
main()
 
# 运行结果：PI: 3.14
```

现在，我们写一个用于计算圆面积的area.py文件，area.py文件需要用到const.py文件中的PI变量。从const.py中，我们把PI变量导入area.py：

```python
from const import PI
 
def calc_round_area(radius):
    return PI * (radius ** 2)
 
def main():
    print("round area: ", calc_round_area(2))
 
main()
 
'''
运行结果：
PI: 3.14
round area:  12.56
'''
```

我们看到 const.py 中的 `main` 函数也被运行了，实际上我们不希望它被运行，因为 const.py 提供的 `main` 函数只是为了测试常量定义。这时 `if __name__ == '__main__'` 派上了用场，我们把const.py改一下：

```python
PI = 3.14
 
def main():
    print("PI:", PI)
 
if __name__ == "__main__":
    main()
```

运行const.py，输出如下：

```python
PI: 3.14
```

运行area.py，输出如下：

```python
round area:  12.56
```

如上，我们可以看到 `if __name__ == '__main__'` 相当于Python模拟的程序入口，Python本身并没有这么规定，这只是一种编码习惯。由于模块之间相互引用，不同模块可能有这样的定义，而程序入口只有一个。到底哪个程序入口被选中，这取决于 `__name__` 的值。

# \__init__(self, ...)

定义类的时候，若是添加 `__init__` 方法，那么在创建类的实例的时候，实例会自动调用这个方法，一般用来对实例的属性进行初使化。比如：

```python
class Student:
    def  __init__(self, name, gender):
        self.name = name
        self.gender = gender

XiaoMing = Student(name='XiaoMing', gender='Male')
print(XiaoMing.name, XiaoMing.gender)
```

 此处，类进行初始化时就会自动调用 `__init__` 中的代码，对 `self.name` 和 `self.gender` 赋值，之后可以通过 `XiaoMing.name` 等来访问。

`self` 是个对象（Object），是当前类的实例。在类的代码（函数）中，需要访问当前的实例中的变量和函数的，即：

- 访问对应变量（property)：Instance.ProperyNam，去读取之前的值和写入新的值

- 调用对应函数（function）：Instance.function()，即执行对应的动作

新建的实例本身，连带其中的参数，会一并传给 `__init__` 函数自动并执行它。所以 `__init__` 函数的参数列表会在开头多出一个参数，它永远指代新建的那个实例对象，Python语法要求函数的第一个参数**必须**是实例对象本身，而名称随意，习惯上就命为 `self`。当然，如果你非要写为别的比如 `me` ，也不是不可以，之后的 `self.xxx` 就要写成 `me.xxx` 。你开心就好。

注意以下三点

- `__init__` 具备**独立的命名空间**，也就是说**函数内新引入的变量均为局部变量**，新建的实例对象对这个函数来说也只是通过第一参数self从外部传入的，故无论设置还是使用它的属性都得利用 `self.<属性名>`。如果将上面的初始化语句中新增`myname = name`（ `myname` 没有用 `self.` 修饰），则只是在函数内部创建了一个 `myname` 变量，它在函数执行完就会消失，对新建的实例没有任何影响；
- 与此对应，**`self`的属性名和函数内其他名称（包括参数）也是不冲突的**，所以下面的写法正确而且规范

```python
class Student:
    def  __init__(self, name, gender):
        # self.name是self的属性，单独的name是函数内的局部变量，参数也是局部变量
        self.name = name
```

- 返回时只能 `return`，不允许带返回值。

# super(XXX, self).\__init__()

这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。当然，如果初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的。

特别是在使用 PyTorch 时，自定义网络类（比如类名为 `CNN` ）一般需要继承 PyTorch 提供的 `nn.Module` ：

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
```



# 参考文献

无。
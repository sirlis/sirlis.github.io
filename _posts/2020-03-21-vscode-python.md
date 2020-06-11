---
layout: post
title:  "VSCode部署Python开发环境"
date:   2020-03-21 15:22:19
categories: Coding
tags: Python
---

# 目录

* [目录](#目录)
* [简介](#简介)
* [VSCode下载与安装](#VSCode下载与安装)
* [配置Python开发环境](#配置Python开发环境)
  * [部署解释器](#部署解释器)
  * [安装Anaconda](#安装Anaconda)
  * [新建和备份环境](#新建和备份环境)
  * [配置依赖包](#配置依赖包)
    * [更新包管理工具](#更新包管理工具)
      * [pip](#pip)
      * [conda](#conda)
      * [说明](#说明)
    * [更换镜像源](#更换镜像源)
      * [pip镜像源](#pip镜像源)
      * [conda镜像源](#conda镜像源)
    * [常用命令](#常用命令)
  * [生成配置文件](#生成配置文件)
    * [解释配置（settings.json）](#解释配置（settings.json）)
    * [调试配置（launch.json）](#调试配置（launch.json）)
  * [调试运行测试](#调试运行测试)
* [错误集合](#错误集合)
  * [无法将conda项识别为cmdet...](#无法将conda项识别为cmdet...)
  * [提示CommandNotFoundError](#提示CommandNotFoundError)
* [参考文献](#参考文献)

# 简介

Python是一种跨平台的计算机程序设计语言。 是一个高层次的结合了解释性、编译性、互动性和面向对象的脚本语言。最初被设计用于编写自动化脚本(shell)，随着版本的不断更新和语言新功能的添加，越多被用于独立的、大型项目的开发。

# VSCode下载与安装

前往官网（https://code.visualstudio.com/）下载安装，支持Windows、Linux和Mac系统。可以下载安装版，也可以选择解压即用的绿色版。区别在于安装板会向系统路径写入配置信息，绿色版所有的依赖信息和配置信息均存放于一个目录中。安装版可以在线下载更新和安装更新，绿色版只能下载新版本的绿色安装包解压后覆盖来更新。

安装完成后，点击左侧的扩展商店，搜索chinese，下载中文简体汉化包（可能需要翻墙）。

![汉化](..\assets\img\postsimg\20200318\01.chinese.png)

安装完成后重启VSCode，即可发现所有界面均已汉化。

注意：

- VSCode基于文件夹进行编译和调试，每个项目必须对应一个文件夹作为工作路径（根目录），根目录内包含一个.vscode文件夹存放配置文件（json格式）；

- VSCode默认编码为UTF8，对中文支持并不完美，特别是打开已有的包含中文注释的源代码文件时要特别注意，可能导致中文乱码，且在保存文件时弹出警告。因此，对于包含中文注释的已有文件，一般需要新建一个空白文件，保存为UTF8编码格式，然后重新输入中文注释部分再进行保存；

- （未完待续）。

# 配置Python开发环境

## 部署解释器

扩展商店搜索“python”，安装微软官方出品的Python扩展。然后重启VSCode。

![汉化](..\assets\img\postsimg\20200321\01.python.png)

然后配置Python解释器。

## 安装Anaconda

推荐采用**Anaconda**配置Python环境，可参考[[1](#ref1)]。Anaconda是一个方便的python包管理和环境管理软件，一般用来配置不同的项目环境。我们常常会遇到这样的情况，正在做的项目A和项目B分别基于python2和python3，而电脑一般只能安装一个环境，这个时候Anaconda就派上了用场，它可以创建多个互不干扰的环境，分别运行不同版本的软件包，以达到兼容的目的。

Anaconda通过管理工具包、开发环境、Python版本，大大简化了你的工作流程。不仅可以方便地安装、更新、卸载工具包，而且安装时能自动安装相应的依赖包，同时还能使用不同的虚拟环境隔离不同要求的项目。

注意，如果全程采用官方网站进行下载安装，**翻墙**是必不可少的。

Anaconda支持Windows、Linux和Mac平台，从官方网站（https://www.anaconda.com/）选择对应平台的Anaconda3安装包下载。对于Windows 10平台，推荐下载64-Bit的Python3.7版本的安装包（614.3M）。

![Anaconda](..\assets\img\postsimg\20200321\02.anacondadownload.png)

采用默认配置安装Anaconda，安装路径可以自定义，但务必记住安装路径。安装无需联网，保险起见请关闭杀毒软件。

进行到下图步骤时，均不勾选，后面我们将手动配置环境变量。

![Anaconda](..\assets\img\postsimg\20200321\03.anacondainstall.png)



## 新建和备份环境

新建环境需要联网。在开始菜单中找到“Anaconda Navigator”，单击打开后，点击左侧的“Environments”，可以看到默认存在一个 `base(root)` 环境。点击下方的Create按钮新建一个环境。

![Anaconda](..\assets\img\postsimg\20200321\04.newenvironment.png)

弹出 `Create new environment` 界面，输入 `Name` ，勾选 `Python` 并下拉选择 `3.7` 版本。然后记住 `Location` 对应的环境路径。最后点击 `Create` 完成环境配置，新建的环境会显示在环境列表中。

![Anaconda](..\assets\img\postsimg\20200321\05.configenv.png)

**注意**，envs 内的每一个子文件夹都是一个独立的环境，删除、重命名子文件夹等价于删除、重命名环境。将子文件夹复制到其他机器的Anaconda的envs文件夹中，该机器的Anaconda可直接识别并应用该环境。因此可在配置好一个环境后，对该子文件夹进行备份。

**激活**某个环境的方法为左键单击该环境。打开某个环境的**终端**为点击环境名称旁边的三角按钮，在弹出菜单中选择 `Open Terminal`。

## 配置依赖包

### 更新包管理工具

#### pip

Python默认的包管理工具是**pip**。输入以下命令查看pip版本

```
pip show pip
```

如果pip版本不是最新的，很多包可能安装不上。可通过命令更新pip。打开环境的终端（Terminal），输入以下命令后回车

```
python -m pip install --upgrade pip
```

如果pip版本不是最新的，会更新到最新版本，如下图所示

![Anaconda](..\assets\img\postsimg\20200321\06.updatepip.png)

如果pip版本已经是最新的，会如下图提示

![Anaconda](..\assets\img\postsimg\20200321\07.piplatest.png)

#### conda

还可以采用第三方的开源跨平台包管理工具**conda**进行包管理，作为pip无法进行包更新时的备份工具。

Anaconda安装后一般默认安装了conda工具。要查看环境是否安装了conda，打开环境的终端，输入

```
conda -V
```

若返回conda的版本号，则表示环境中默认搭载了conda。

![Anaconda](..\assets\img\postsimg\20200321\08.condaversion.png)

输入以下命令更新conda

```
conda update -n base conda
```

期间提示是否更新包，输入y确认

![Anaconda](..\assets\img\postsimg\20200321\13.updateconda.png)

#### 说明

Conda和pip通常被认为几乎完全相同。虽然这两个工具的某些功能重叠，但它们设计用于不同的目的。 [Pip](https://pip.pypa.io/en/stable/)是Python Packaging Authority推荐的用于从[Python Package Index](https://pypi.org/)安装包的工具。 Pip安装打包为wheels或源代码分发的Python软件。后者可能要求系统安装兼容的编译器和库。

[Conda](https://conda.io/docs/)是跨平台的包和环境管理器，可以安装和管理来自[Anaconda repository](https://repo.anaconda.com/)以 [Anaconda Cloud](https://anaconda.org/)的conda包。 Conda包是二进制文件，徐需要使用编译器来安装它们。另外，conda包不仅限于Python软件。它们还可能包含C或C ++库，R包或任何其他软件。

这是conda和pip之间的关键区别。 Pip安装Python包，而conda安装包可能包含用任何语言编写的软件的包。**在使用pip之前，必须通过系统包管理器或下载并运行安装程序来安装Python解释器。而Conda可以直接安装Python包以及Python解释器**，即conda将python本身也当做一个包来管理。

另外，conda查看环境中安装的所有包时，可以包含从Anaconda界面安装的包，而pip则只能查看到所有通过命令行安装的包。如下图所示，通过Anaconda界面安装的cudatoolkit和cudnn包，在pip中无法查到。

![不同包管理工具的查看区别](..\assets\img\postsimg\20200322\08.different.png)

对于用户而言，尽可能从一而终的采用一种包管理工具。若使用Anaconda配置的python环境，则推荐使用conda，配合Anaconda界面使用更加友好，除非某些包无法通过conda安装，则可采用pip安装。

### 更换镜像源

#### pip镜像源

pip的默认镜像源在国外，更新包会下载缓慢甚至无法下载，可更换到国内的镜像源（清华、阿里、中科大等）。

对于Linux系统，直接修改 `~/.pip/pip.conf` (没有就创建一个)， 内容如下

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

对于Windows系统，快捷键 `win+R`  打开运行，输入 `%HOMEPATH%` 回车打开用户目录，在此目录下创建 pip 文件夹，在 pip 文件夹内创建 pip.ini 文件, 内容如下

```
[global]
timeout = 6000
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

点击 “Update index...” 更新索引

![更新索引](..\assets\img\postsimg\20200321\09.updateindex.png)

#### conda镜像源

可通过修改用户目录下的 `.condarc` 文件来更换源，文件位于

```
Windows：C:\Users\xxx\.condarc
Linux：/home/xxx/.condarc
```

其中 `xxx` 为用户账户名称。如果该路径下没有该文件，可自行新建一个，注意文件全名为 “`.condarc`”，没有其它任何后缀。若Windows用户无法直接新建该文件，可以先执行以下命令生成该文件

```
conda config --set show_channel_urls yes
```

将下面的代码覆盖文件中所有内容，完成源更换

```
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

运行一下命令清除缓存

```
conda clean -i
```

点击 “Update index...” 更新索引

![更新索引](..\assets\img\postsimg\20200321\09.updateindex.png)



### 常用命令

- 显示环境下所有安装的包

```
[pip]    pip -v list
[conda]  conda list 
```

![piplist](..\assets\img\postsimg\20200321\10.piplist.png)

- 显示所有过时包

```
pip list --outdated
```

![piplist](..\assets\img\postsimg\20200321\11.pipoutdated.png)

- 更新特定包

```
pip install --upgrade xxxx
```

![piplist](..\assets\img\postsimg\20200321\12.pipupgrade.png)

- 更新pip

```
python -m pip install --upgrade pip
```

- 更新conda

```
conda update -n base -c defaults conda
```

## 生成配置文件

在项目工作路径下新建 `.vscode` 文件夹，其中新建以下两个配置文件，并用下面的内容填充。

### 解释配置（settings.json）

```
{
    "python.pythonPath": "E:\\ProgramFiles\\Anaconda3\\envs\\Pytorch\\python.exe"
}
```

其中具体的python路径位置因Anaconda安装位置不同而不同，注意转义字符 `\\`。

### 调试配置（launch.json）

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 当前文件",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

## 调试运行测试

随便新建一个 `python` 文件，如 `printtest.py` 进行测试，下图可以看出具备代码智能补全功能。

![调试运行测试](..\assets\img\postsimg\20200321\15.test.png)

按 `F5` 运行结果如下

![调试运行测试](..\assets\img\postsimg\20200321\16.run.png)

# 错误集合 

 ## 无法将conda项识别为cmdet...

VSCode 解释 `.py` 时，终端自动运行命令 `conda activate Pytorch(环境名)` 时提示

```
无法将“conda”项识别为 cmdlet、函数、脚本文件或可运行程序的名称。
```

解决方法如下。

要么将Anaconda安装路径的三个路径变量写入系统Path中

```
D:\XXX\Anaconda3
D:\XXX\Anaconda3\Scripts
D:\XXX\Anaconda3\Library\bin
```

要么打开VSCode的设置（`ctrl+,`），设置Python插件中的conda的路径

![condapath](..\assets\img\postsimg\20200321\14.condapath.png)



## 提示CommandNotFoundError

VSCode 解释 `.py` 时，终端自动运行命令 `conda activate xxx(环境名)` 时提示

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
If using 'conda activate' from a batch script, change your
invocation to 'CALL conda.bat activate'.
```

解决方法如下。

首先在VSCode 的终端中输入 `conda init`，然后重启VSCode。

此时会出现错误，`无法加载文件 ******.ps1，因为在此系统中禁止执行脚本。`

关闭VSCode。按 `Win+X`，选择 `Windows PowerShell（管理员）` 打开，输入命令 `set-ExecutionPolicy RemoteSigned` 回车

```
执行策略更改 
执行策略可以防止您执行不信任的脚本。更改执行策略可能会使您面临 about_Execution_Policies 
帮助主题中所述的安全风险。是否要更改执行策略? 
[Y] 是(Y)  [N] 否(N)  [S] 挂起(S)  [?] 帮助 (默认值为“Y”): y
```

然后输入 `y` 回车。

之后重启VSCode，`F5` 执行Python文件应该就不会提示错误了。

# 参考文献

<span id="ref1">[1]</span> [挖掘机小王子](https://www.zhihu.com/people/WaJueJiPrince). [VSCode+Anaconda打造舒适的Python环境](https://zhuanlan.zhihu.com/p/30324113).

<span id="ref1">[2]</span> [Eric-Young](https://www.cnblogs.com/Eric-Young/). [python之VSCode安装](https://www.cnblogs.com/Eric-Young/p/6393513.html).

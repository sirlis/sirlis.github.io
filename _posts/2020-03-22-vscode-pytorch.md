---
title: VSCode部署Pytorch机器学习框架
date: 2020-03-22 09:22:19 +0800
categories: [Coding]
tags: [python]
math: true
---

# 简介

PyTorch是一个开源的Python机器学习库，基于Torch，用于自然语言处理等应用程序。2017年1月，由Facebook人工智能研究院（FAIR）基于Torch推出。它是一个基于Python的可续计算包，提供两个高级功能：1、具有强大的GPU加速的张量计算（类似NumPy）。2、包含自动求导系统的的深度神经网络。

吃别人一记强力安利：[PyTorch到底好用在哪里？](https://www.zhihu.com/question/65578911)

# 配置Python开发环境

参考[《VSCode部署Python开发环境》](http://sirlis.github.io/2020-03-21-vscode-python/)配置。

# 配置PyTorch

## 部署PyTorch

前往官网（https://pytorch.org/get-started/locally/），根据自身的开发环境，获取PyTorch安装命令行。以Windows 10系统+RTX2060显卡为例，采用pip安装，如图选择，得到安装命令行

![安装命令行](\assets\img\postsimg\20200322\01.command.png)

**注意**，PyTorch包含两个版本，CPU版（CUDA=None）和GPU版，若计算机没有合适的独立显卡，则CUDA选择None。不过GPU版同样包含CPU版的所有功能，因此完全可以安装GPU版，只是不能利用GPU计算加速。

**注意**，请自行确认独立显卡驱动支持的**CUDA版本**。打开控制面板，选择查看方式为“小图标”，选择“Nvidia控制面板”，然后如图所示的步骤依次打开“系统信息” => “组件”，查看 “NVCUDA.DLL” 的产品名称，并选择不超过其版本号的CUDA版本号。

![CUDA版本查看](\assets\img\postsimg\20200322\04.cudaversion.png)

若采用pip安装，则命令行如下

```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```

若采用conda安装，则命令行如下

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

若已经更换了 Anaconda 的镜像源为国内源，则可以去掉后面的 `-c pytorch`，即使用

```
conda install pytorch torchvision cudatoolkit=10.1
```

打开Anaconda Navigator，激活相应的环境，打开环境的终端，输入上述命令即可完成PyTorch的安装。

完整的GPU版本的PyTorch包含以下组件，版本号为我使用的版本号，其他人需要根据自身实际情况调整：

- pytorch  1.5.0
- torchvision  0.6.0
- cuda  10.2.89（后文中部署）
- cudnn  7.6.5（后文中部署）

## 部署其它包

### CUDA

CUDA（Compute Unified Device Architecture），是NVIDIA推出的运算平台。 CUDA™是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。 它包含了CUDA指令集架构（ISA）以及GPU内部的并行计算引擎。 开发人员可以使用C语言来为CUDA™架构编写程序。所编写出的程序可以在支持CUDA™的处理器上以超高性能运行。

CUDA依赖显卡驱动，提前更新显卡驱动并确认显卡驱动支持的CUDA版本号。

采用命令行安装时，命令行中已经带有安装CUDA的指令 `cudatoolkit=10.1`。若命令行安装失败，可通过Anaconda界面依次安装pytorch、torchvision和cudatoolkit。

若界面安装仍然失败，可尝试手动安装，请前往 [手动部署CUDA和cuDNN](#手动部署CUDA和cuDNN)。

### cuDNN

可通过Anaconda界面安装。

若界面安装失败，可尝试手动安装，请前往 [手动部署CUDA和cuDNN](#手动部署CUDA和cuDNN)。

### Numpy

NumPy（Numerical Python）是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多（该结构也可以用来表示矩阵（matrix）），支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

- 注意！numpy若通过 conda 或 Anaconda 界面安装可能存在问题，需要用 pip 安装
- 注意！基础包一定要先安装并测试好能用，再安装其他包。比如先装numpy，再安装scipy，matplotlib等。

采用命令行安装

```
pip install numpy
conda install numpy
```

或者通过Anaconda界面进行安装。

### matplotlib

matplotlib包用以进行绘图，采用命令行安装

```
conda install matplotlib
```

或者通过Anaconda界面进行安装

![安装matplotlib](\assets\img\postsimg\20200322\02.matplotlib.png)

### pandas

pandas包用于输入输出和处理csv格式数据，采用命令行安装

```
conda install pandas
```

或者通过Anaconda界面进行安装

![安装matplotlib](\assets\img\postsimg\20200322\03.pandas.png)

## 手动部署CUDA和cuDNN

若自动安装CUDA和cuDNN失败，也可选择手动安装部署CUDA。

首先需要更新自己的显卡驱动，此处不再赘述。

若要手动部署CUDA和cuDNN，必须遵循先CUDA后cuDNN的顺序。首先前往官网（https://www.nvidia.com/）下载CUDA。

![安装matplotlib](\assets\img\postsimg\20200322\05.manualcuda.png)

在打开的页面中点击 ”Download Now“ 按钮，然后再新页面中选择 “Legacy Releases” 按钮，不要按照页面的说法进行系统选择等操作。

![安装matplotlib](\assets\img\postsimg\20200322\06.cuda1.png)

然后根据自己的实际情况选择相应的CUDA版本下载安装。

![安装matplotlib](\assets\img\postsimg\20200322\07.cuda2.png)

手动安装CUDA后需要进行检查。`win+R` 输入 `cmd` 回车，打开命令提示符，输入

```
nvcc -V
```

若成功返回cuda版本等信息则表示安装成功。

![CUDA版本](\assets\img\postsimg\20200322\09.cudaversion.png)

继续输入（其中路径自行根据CUDA安装路径调整）

```
cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\demo_suite
```

然后输入 `deviceQuery.exe`，执行此程序。出现 `PASS` 即表示CUDA安装成功。

![CUDA版本](\assets\img\postsimg\20200322\10.cudapass.png)

然后，前往[此处](https://developer.nvidia.com/cudnn)（https://developer.nvidia.com/cudnn），点击 “Download cuDNN” 按钮下载cuDNN。下载前需要书册账号并登陆。**注意**，cuDNN版本与CUDA版本间存在匹配关系，下载时一定要注意。

下载解压后得到的文件直接覆盖到CUDA安装路径，如下图所示。

![CUDNN安装](\assets\img\postsimg\20200322\11.cudnn.png)

## 测试

在环境中启动终端，输入

```
python
```

启动python环境。

![12.test1](\assets\img\postsimg\20200322\12.test1.png)

然后一行行输入以下命令

```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

上面的代码用于产生一个5行3列的随机矩阵（张量），输出应该为下面类似的形式

```python
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

![13.test2](\assets\img\postsimg\20200322\13.test2.png)

检查pytorch是否能够正确调用GPU驱动和是否能够启用CUDA，输入：

```python
import torch
torch.cuda.is_available()
```

返回 `True` 即可。

![14.test3](\assets\img\postsimg\20200322\14.test3.png)

# 参考文献

<span id="ref1">[1]</span> [Sunnyside_Bao](https://blog.csdn.net/Sunnnyside_Bao). [Anaconda＋vscode＋pytorch环境搭建](https://blog.csdn.net/Sunnnyside_Bao/article/details/93495605).


---
title: 强化学习
date: 2022-11-09 112:36:19 +0800
categories: [Academic, Knowledge]
tags: [python]
math: true
---

本文介绍了强化学习的基本概念和模型。

<!--more-->

---

- [1. 基本知识](#1-基本知识)
  - [1.1. 强化学习](#11-强化学习)
  - [1.2. assimp](#12-assimp)
- [2. 部署方式](#2-部署方式)
  - [2.1. CMake](#21-cmake)
  - [2.2. make](#22-make)
- [3. 参考文献](#3-参考文献)

## 1. 基本知识

### 1.1. 强化学习

强化学习是机器学习领域之一，受到行为心理学的启发，主要关注智能体如何在环境中采取不同的行动，以最大限度地提高累积奖励。

强化学习主要由智能体（Agent）、环境（Environment）、状态（State）、动作（Action）、奖励（Reward）组成。智能体执行了某个动作后，环境将会转换到一个新的状态，对于该新的状态环境会给出奖励信号（正奖励或者负奖励）。随后，智能体根据新的状态和环境反馈的奖励，按照一定的策略执行新的动作。上述过程为智能体和环境通过状态、动作、奖励进行交互的方式。

假设体当前时刻 $t$ 智能体所处的状态记为 $s_t$，此时采取动作 $a_t$，改变了环境状态，并使得智能体在下一时刻 $t+1$ 达到了新状态 $s_{t+1}$，在新的状态下环境产生了反馈奖励 $r_{t+1}$ 给智能体。智能体根据新状态 $s_{t+1}$ 和反馈奖励 $r_{t+1}$ ，执行新动作 $a_{t+1}$，如此反复迭代交互。

![强化学习示意图](/assets/img/postsimg/20221109/0-reinforcement-learning-basic-diagram.jpg)

上述过程的最终目的，是让智能体最大化累计奖励（Cumulative Award），累计奖励为 $G$，有

$$
G = r_1+r_2+...+r_n
$$

### 1.2. assimp

Assimp 全称为 Open Asset Import Library，这是一个模型加载库，可以导入几十种不同格式的模型文件（同样也可以导出部分模型格式）。只要 Assimp 加载完了模型文件，我们就可以从 Assimp 上获取所有我们需要的模型数据。Assimp 把不同的模型文件都转换为一个统一的数据结构，所有无论我们导入何种格式的模型文件，都可以用同一个方式去访问我们需要的模型数据。

官方手册地址：https://assimp-docs.readthedocs.io/en/v5.1.0/

官方仓库地址：https://github.com/assimp/assimp

Assimp 基本上没有预编译的文件，而且为了适配本机环境，最好还是自己编译，因此我们需要下载 Assimp 的源码。

## 2. 部署方式

部署过程在如下版本部署成功：

- assimp 5.2.5
- OS: Windows 11
- CMake 3.25.0-rc2
- gcc version 12.2.0 (x86_64-win32-sjlj-rev0, Built by MinGW-W64 project)
- MinGW-w64: https://github.com/niXman/mingw-builds-binaries/releases 下载的 2022 Aug 23 版本

### 2.1. CMake

首先需要下载 CMake，官网：https://cmake.org/

下载完成后运行 CMake(cmake-gui)，设置源代码路径（where is the source code）和二进制文件路径（where to build the binaries）

![配置CMake](/assets/img/postsimg/20221026/config_path.png)


点击 `Configure` 按钮进行配置。配置生成所需的 Makefiles

![配置CMake generator](/assets/img/postsimg/20221026/specify_generator.png)

配置编译器

![配置CMake compilers](/assets/img/postsimg/20221026/config_compilers.png)

**注意：** 完成配置后，取消勾选 `ASSIMP_WARNINGS_AS_ERRORS`，否则会将 Warning 看作 Error 报错。

![配置CMake完成](/assets/img/postsimg/20221026/error_as_warning_and_generate.png)

最后点击 `Generate` 按钮生成文件和 `makefile`。

### 2.2. make

管理员打开 PowerShell 或者命令提示符，cd 到设置的二进制文件路径，运行下面的命令

```cmd
mingw32-make.exe -j8
```

注意：前提是 MinGW-w64 安装路径下的 `bin` 文件夹已经添加到系统的环境变量（PATH）中。

注意：`-j8` 表示使用 CPU 的八核进行编译，根据自己的硬件情况设置。

![make完成](/assets/img/postsimg/20221026/cmake_making.png)

编译完成后，得到

- include/assimp/config.h
- bin/libassimp-5.dll
- bin/unit.exe
- lib/libassimp.dll.a

## 3. 参考文献

无。
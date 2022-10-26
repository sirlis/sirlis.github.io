---
title: Windows环境下使用CMake+MinGW-w64编译模型加载库assimp
date: 2022-09-23 16:23:19 +0800
categories: [Tutorial, Coding]
tags: [c]
math: true
---

本文介绍了在Windows环境下使用MinGW-w64编译模型加载库assimp的方法和坑。

<!--more-->

---

- [1. 基本知识](#1-基本知识)
  - [1.1. MinGW-w64](#11-mingw-w64)
  - [1.2. assimp](#12-assimp)
- [2. 部署方式](#2-部署方式)
  - [2.1. CMake](#21-cmake)
  - [2.2. make](#22-make)
- [3. 参考文献](#3-参考文献)

## 1. 基本知识

### 1.1. MinGW-w64

MinGW 的全称是 Minimalist GNU on Windows 。是将经典的开源 C 语言编译器 GCC 移植到了 Windows 平台下，并且包含了 Win32API ，因此可以将源代码编译为可在 Windows 中运行的可执行程序。而且还可以使用一些 Windows 不具备的，Linux平台下的开发工具。

**一句话来概括：MinGW 就是 GCC 的 Windows 版本 。**

MinGW-w64 与 MinGW 的区别在于 MinGW 只能编译生成32位可执行程序，而 MinGW-w64 则可以编译生成 64位 或 32位 可执行程序。正因为如此，MinGW 现已被 MinGW-w64 所取代，且 MinGW 也早已停止了更新，内置的 GCC 停滞在了 4.8.1 版本，而 MinGW-w64 内置的 GCC 则一直保持更新。

可在[此处](https://github.com/niXman/mingw-builds-binaries/releases)下载最新版本。(https://github.com/niXman/mingw-builds-binaries/releases)。

更多介绍可参考：[VSCode部署C/C++开发环境](http://sirlis.cn/posts/vscode-c/)。

### 1.2. assimp

Assimp 全称为 Open Asset Import Library，这是一个模型加载库，可以导入几十种不同格式的模型文件（同样也可以导出部分模型格式）。只要 Assimp 加载完了模型文件，我们就可以从 Assimp 上获取所有我们需要的模型数据。Assimp 把不同的模型文件都转换为一个统一的数据结构，所有无论我们导入何种格式的模型文件，都可以用同一个方式去访问我们需要的模型数据。

官方手册地址：https://assimp-docs.readthedocs.io/en/v5.1.0/

官方仓库地址：https://github.com/assimp/assimp

## 2. 部署方式

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


## 3. 参考文献

无。
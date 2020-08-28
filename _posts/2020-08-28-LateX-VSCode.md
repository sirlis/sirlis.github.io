---
layout: post
title:  "LaTeX+VSCode环境配置"
date:   2020-08-28 09:17:19
categories: Coding
tags: Latex
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
* [前言](#前言)
* [LaTeX配置](#LaTeX配置)
  * [安装MikTeX](#安装MikTeX)
  * [配置环境变量](#配置环境变量)
* [配置VSCode的LaTeX环境](#配置VSCode的LaTeX环境)
  * [安装LaTeX Workshop](#安装LaTeX Workshop)
  * [配置json](#配置json)
  * [编译测试](#编译测试)
* [参考文献](#参考文献)

# 前言

由于VSCode太牛逼，所有的C和Python仿真均已经迁移至该编辑器下完成，偶然发现其还可编译LaTeX，狂喜，遂研究之，步骤列于下。

下面以 MikTeX 20.6 + VSCode 1.48.2 为例进行安装和部署讲解。

# LaTeX配置

## 安装MikTeX

参考[下载和安装MikTeX](./2020-07-20-LateX.md)。此处摘录如下：

[官网的下载页面](https://miktex.org/download)（https://miktex.org/download）包括三种下载（安装）方式，如图分别为安装程序（Installer）、绿色版（Portable Edition）以及命令行（Command-line installer）。对于Windows开发环境，不考虑命令行方式，因此可以任意选择安装程序或者绿色版。

![image-20200720222610238](..\assets\img\postsimg\20200720\1.jpg)

需要注意的是，绿色版并没有单独的压缩包，而是直接对应安装版的安装程序，只不过将安装程序重命名为 `miktex-portable.exe`，然后双击安装即可。绿色版与安装版的区别在于，绿色版不会向系统盘写入配置信息，也不会注册环境变量，意味着之后如果需要安装编辑器，无法自动获取系统中已经安装的LaTeX版本，而需要手动配置。

双击下载的 exe 文件进行安装，路径任意。

## 配置环境变量

将 miktex 附带的 `xelatex.exe` 和 `pdflatex.exe` 等工具所在的路径加入系统 Path 环境变量。对于绿色版，该路径为

```
X:\ProgramFiles\MikTexPortable\texmfs\install\miktex\bin\x64
```

**注意**，如果在 VSCode 打开的情况下改变了环境变量，需要重启 VSCode 使其能够获取最新的环境变量。

# 配置VSCode的LaTeX环境

## 安装LaTeX Workshop

LaTeX Workshop 几乎可以认为是 VSCode 标配的 LaTeX 编译扩展，挂上翻墙通过扩展商店搜索 latex 弹出的第一个就是。

![image-20200828095040527](..\assets\img\postsimg\20200828\1.jpg)

安装完成后，`ctrl+,` 打开设置面板（或通过左下角的小齿轮点击进入），搜索 `json` 然后点击 「在settings.json 中编辑」，打开 settings.json。

## 配置json

在 `settings.json` 中新增如下代码：

```json
{
    "latex-workshop.latex.autoBuild.run": "never",
	"latex-workshop.latex.recipes": [
      {
        "name": "pdflatex -> bibtex -> pdflatex*2",
        "tools": [
          "pdflatex",
          "bibtex",
          "pdflatex",
          "pdflatex"
        ]
      }
    ],
    "latex-workshop.latex.tools": [
      {
        "name": "xelatex",
        "command": "xelatex",
        "args": [
          "-synctex=1",
          "-interaction=nonstopmode",
          "-file-line-error",
          "%DOC%"
        ]
      },
      {
        "name": "latexmk",
        "command": "latexmk",
        "args": [
          "-synctex=1",
          "-interaction=nonstopmode",
          "-file-line-error",
          "%DOC%"
        ]
      },
      {
        "name": "pdflatex",
        "command": "pdflatex",
        "args": [
          "-synctex=1",
          "-interaction=nonstopmode",
          "-file-line-error",
          "%DOC%"
        ]
      },
      {
        "name": "bibtex",
        "command": "bibtex",
        "args": [
          "%DOCFILE%"
        ]
      }
    ],
    "latex-workshop.latex.clean.fileTypes": [  //设定清理文件的类型  
      "*.aux",  
      "*.bbl",  
      "*.blg",  
      "*.idx",  
      "*.ind",  
      "*.lof",  
      "*.lot",  
      "*.out",  
      "*.toc",  
      "*.acn",  
      "*.acr",  
      "*.alg",  
      "*.glg",  
      "*.glo",  
      "*.gls",  
      "*.ist",  
      "*.fls",  
      "*.log",  
      "*.fdb_latexmk",  
      "*.nav",  
      "*.snm",  
      "*.synctex.gz"  
    ],
    "latex-workshop.view.pdf.viewer": "tab",
}
```

最终效果如下（忽略前面的若干主体配置项）：

![image-20200828095421375](..\assets\img\postsimg\20200828\2.jpg)

**注意**，`latex-workshop.latex.tools` 字段定义了编译 LaTeX 的序列操作，默认为 `xelatex -> bibtex -> xelatex*2`，这里将其修改为 `pdflatex -> bibtex -> pdflatex*2`，对应的顺序为调用1次 `pdflatex`，1次 `bibtex`，2次 `pdflatex`，**与texstudio保持一致，确保生成的 pdf 文件字体和格式一致**。

## 编译测试

快捷键 `ctrl+alt+B` 编译 .tex文件，快捷键 `ctrl+alt+v` 或者右上角的「查看pdf」图标查看 .pdf 文件。

![image-20200828101128126](..\assets\img\postsimg\20200828\3.jpg)

# 参考文献

<span id="ref1">[1]</span>  [当年老王](https://blog.csdn.net/yinqingwang). [论文写作的又一利器：VSCode + Latex Workshop + MikTex + Git](https://blog.csdn.net/yinqingwang/article/details/79684419).

<span id="ref2">[2]</span>  [strange_jiong](https://blog.csdn.net/dream_allday). [Latex编译出现字体获取不到的情况](https://blog.csdn.net/dream_allday/article/details/84997874).

<span id="ref3">[3]</span>  [开心鲨鱼](https://www.zhihu.com/people/kai-xin-sha-yu). [配置VScode编辑LaTeX及正反向搜索等设置](https://zhuanlan.zhihu.com/p/90526218?utm_source=wechat_session).
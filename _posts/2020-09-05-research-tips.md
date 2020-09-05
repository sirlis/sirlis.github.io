---
layout: post
title:  "科研Tips"
date:   2020-09-05 14:23:19
categories: Researching
tags: Others
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
* [arXiv检索和下载](#arXiv检索和下载)
* [参考文献](#参考文献)

# arXiv检索和下载

参考：[如何快速下载 arxiv 论文](https://www.jianshu.com/p/184799230f20)

**arXiv**（*X*依希腊文的χ发音，读音如英语的*archive*）（https://arxiv.org/）是一个收集物理学、数学、计算机科学、生物学与数理经济学的论文预印本（preprint）的网站，始于1991年8月14日。截至2008年10月，arXiv已收集超过50万篇预印本[[2\]](https://zh.wikipedia.org/wiki/ArXiv#cite_note-2)[[3\]](https://zh.wikipedia.org/wiki/ArXiv#cite_note-3)；至2014年底，藏量达到1百万篇[[4\]](https://zh.wikipedia.org/wiki/ArXiv#cite_note-4)[[5\]](https://zh.wikipedia.org/wiki/ArXiv#cite_note-dddmag2015-5)。截至2016年10月，提交率已达每月超过10,000篇[[5\]](https://zh.wikipedia.org/wiki/ArXiv#cite_note-dddmag2015-5)[[6\]](https://zh.wikipedia.org/wiki/ArXiv#cite_note-6)。

arxiv 在中国有官方镜像 [http://cn.arxiv.org](https://links.jianshu.com/go?to=http%3A%2F%2Fcn.arxiv.org%2F)，通过使用 chrome 插件将 arxiv 的链接自动重定向到中国镜像网站链接即可，这样当你点击一篇文章的arxiv链接后就可以自动到cn.arxiv.org，速度很快。如果 http://cn.arxiv.org/ 仍然难以访问，但是中科院理论物理所也有一个备选网址： [http://xxx.itp.ac.cn/](https://links.jianshu.com/go?to=http%3A%2F%2Fxxx.itp.ac.cn%2F) ，但是也不是特别稳定。

首先安装 tampermonkey（油猴插件），这是一款功能强大的脚本插件，可以通过脚本对浏览器上网页进行修改编辑等，更多介绍可以参考 [https://zhuanlan.zhihu.com/p/28869740](https://links.jianshu.com/go?to=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F28869740)
。这里我们使用该插件对网页中的 arxiv 链接进行重定向到 cn.arxiv.org。

油猴插件在各类浏览器中（如Chrome，Microsoft Edge等）均可以安装。推荐使用 Chrome  Webstore 或微软的浏览器插件商城（https://microsoftedge.microsoft.com/addons/Microsoft-Edge-Extensions-Home?hl=zh-CN）下载油猴插件，在 crx4chrome 网站搜索并下载也可以，这里给出crx4chrome网站上tampermonkey插件的[下载链接](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.crx4chrome.com%2Fdown%2F755%2Fcrx%2F)。

然后添加 arxiv 重定向脚本。 代码更新时间2017年12年04日，自动转到pdf链接。 代码需要全部复制粘贴，部分看似注释的代码也有用处，代码如下：

```javascript
// ==UserScript==
// @name        Redirect arxiv.org to CN.arxiv.org/pdf
// @namespace   uso2usom
// @description On any web page it will check if the clicked links goes to arxiv.org. If so, the link will be rewritten to point to cn.arxiv.org
// @include     http://*.*
// @include     https://*.*
// @version     1.2
// @grant       none
// ==/UserScript==

// This is a slightly brute force solution, but there is no other way to do it using only a userscript.

// Release Notes

// version 1.2
// Focus on pdf link only!
// Add '.pdf' link  automatically. Convenient for saving as pdf.

// version 1.1
// Redirect arxiv.org to CN.arxiv.org

document.body.addEventListener('mousedown', function(e){
    var targ = e.target || e.srcElement;
    if ( targ && targ.href && targ.href.match(/https?:\/\/arxiv.org\/pdf/) ) {
        targ.href = targ.href.replace(/https?:\/\/arxiv\.org/, 'http://cn.arxiv.org');
    }
    if ( targ && targ.href && targ.href.match(/http?:\/\/arxiv.org\/pdf/) ) {
        targ.href = targ.href.replace(/http?:\/\/arxiv\.org/, 'http://cn.arxiv.org');
    }
    if ( targ && targ.href && targ.href.match(/https?:\/\/arxiv.org\/abs/) ) {
        targ.href = targ.href.replace(/https?:\/\/arxiv\.org\/abs/, 'http://cn.arxiv.org/pdf');
    }
    if ( targ && targ.href && targ.href.match(/http?:\/\/arxiv.org\/abs/) ) {
        targ.href = targ.href.replace(/http?:\/\/arxiv\.org\/abs/, 'http://cn.arxiv.org/pdf');
    }
    if (targ && targ.href && targ.href.match(/http?:\/\/cn.arxiv.org\/pdf/) && !targ.href.match(/\.pdf/) )
    {
       targ.href = targ.href + '.pdf';
    }
});
```

最后，测试配置是否成功，下面是 arxiv 上的一篇文章，点击看网址前面是否已经加上“cn.”前缀，点击pdf测试速度。该文章共57页，之后可以手动去掉“cn.”前缀对比速度。
 [NIPS 2016 Tutorial: Generative Adversarial Networks](https://links.jianshu.com/go?to=http%3A%2F%2Farxiv.org%2Fabs%2F1701.00160)

另外，由于 [http://cn.arxiv.org](https://links.jianshu.com/go?to=http%3A%2F%2Fcn.arxiv.org%2F) 并不是主站点，是 arxiv 在中国区的镜像，因此更新有大约半天的延迟，对于当天提交的文章，可能更新不及时。对于当天文章可以手动删除“cn.”前缀解决。 如果出现 pdf 正在自动从源文件生成等提示，为正常现象，稍后即可获取pdf论文。

# LaTeX配置

## 安装MikTeX

参考[下载和安装MikTeX](./2020-07-20-LateX.md)。此处摘录如下：

[官网的下载页面](https://miktex.org/download)（https://miktex.org/download）包括三种下载（安装）方式，如图分别为安装程序（Installer）、绿色版（Portable Edition）以及命令行（Command-line installer）。对于Windows开发环境，不考虑命令行方式，因此可以任意选择安装程序或者绿色版。

![image-20200720222610238](..\assets\img\postsimg\20200720\1.jpg)

需要注意的是，绿色版并没有单独的压缩包，而是直接对应安装版的安装程序，只不过将安装程序重命名为 `MiKTeX-portable.exe`，然后双击安装即可。绿色版与安装版的区别在于，绿色版不会向系统盘写入配置信息，也不会注册环境变量，意味着之后如果需要安装编辑器，无法自动获取系统中已经安装的LaTeX版本，而需要手动配置。**懒人推荐安装版，省去配置环境变量等步骤**（虽然后面是以绿色版介绍的）。

双击下载的 exe 文件进行安装，路径任意。

## 配置环境变量（绿色版）

将 miktex 附带的 `xelatex.exe` 和 `pdflatex.exe` 等工具所在的路径加入系统 Path 环境变量。假设安装的MiKTeX为绿色版，安装根目录为`X:\ProgramFiles\MiKTeX\`，则上述路径均位于

```
X:\ProgramFiles\MiKTeX\texmfs\install\miktex\bin\x64
```

相应的，安装版的路径位于

```
X:\ProgramFiles\MiKTeX\miktex\bin\x64
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
    "latex-workshop.view.pdf.viewer": "tab",
}
```

最终效果如下（忽略前面的若干主体配置项）：

![image-20200828095421375](..\assets\img\postsimg\20200828\2.jpg)

**注意**，`latex-workshop.latex.tools` 字段定义了编译 LaTeX 的序列操作，默认为 `xelatex -> bibtex -> xelatex*2`，这里将其修改为 `pdflatex -> bibtex -> pdflatex*2`，对应的顺序为调用1次 `pdflatex`，1次 `bibtex`，2次 `pdflatex`，**与texstudio保持一致，确保生成的 pdf 文件字体和格式一致**。

进阶配置还包括：设置禁止保存时自动build，以及设定自动清理中间文件的类型。

```json
{
    "latex-workshop.latex.autoBuild.run": "never", //禁止保存时自动build  
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
}
```

设置仅针对 LaTeX 的自动换行

```json
{
    "latex-workshop.view.pdf.viewer": "tab",
    "[latex]": {
      "editor.wordWrap": "on", // <== auto wrap
      "editor.formatOnPaste": false,
      "editor.suggestSelection": "recentlyUsedByPrefix"
    },
}
```



## 编译测试

快捷键 `ctrl+alt+B` 编译 .tex文件，快捷键 `ctrl+alt+v` 或者右上角的「查看pdf」图标查看 .pdf 文件。

![image-20200828101128126](..\assets\img\postsimg\20200828\3.jpg)

## **快捷键**

快捷键的更改根据个人习惯而定。

打开键盘快捷方式面板(左下侧齿轮，或快捷键`ctrl+k,ctrl+s`)：

- 搜索"切换侧栏可见性"，设置快捷键为`ctrl+k ctrl+b`。

- 搜索 `latex build`，将默认的`ctrl+alt+b`替换为`ctrl+b`(与Sublime Text 3统一)。

- 搜索`latex recipe`，设置快捷键为`ctlr+r`，方便点菜(选择编译方式)！(ST3中是显示文档大纲)。

- 其他常用的快捷键：

- - `ctrl+k ctrl+a`： 切换活动栏可见性(左侧图标开关)
  - `ctrl+alt+x`：显示LaTeX面板(左侧编译命令面板和文档大纲)。
  - `ctrl+alt+c`：清除辅助文件
  - `ctrl+alt+v`：查看编译的pdf文件(预览)
  - `ctrl+alt+j`：正向搜索。当设置`"latex-workshop.view.pdf.viewer": "tab";`时，在LaTeX源文件中按下快捷键，定位到PDF文档相应位置。(反向搜索见后面)

根据 `latex-workshop.latex.recipes` 中的`name`段设定，可在 `.tex` 文件首行指定编译方式。如 `%!TEX program = xelatex` 表示用 xelatex 编译文件，而 `%!TEX program = PDFlatex` 表示用 latexpdf 编译文件。多个文件情况，还可以用 `% !TEX root` 指定主文件，`% !TEX bib `指定 bib 的编译方式。

**示例**

```text
%! TeX program = pdflatex
\documentclass{article}

\begin{document}
    press ctrl+b to complie，press ctrl+alt+v to view pdf
\end{document}
```

# 参考文献

<span id="ref1">[1]</span> [德谟赛斯](https://www.jianshu.com/u/06ba6c212ceb). [如何快速下载 arxiv 论文](https://www.jianshu.com/p/184799230f20).

<span id="ref2">[2]</span>  [strange_jiong](https://blog.csdn.net/dream_allday). [Latex编译出现字体获取不到的情况](https://blog.csdn.net/dream_allday/article/details/84997874).

<span id="ref3">[3]</span>  [开心鲨鱼](https://www.zhihu.com/people/kai-xin-sha-yu). [配置VScode编辑LaTeX及正反向搜索等设置](https://zhuanlan.zhihu.com/p/90526218?utm_source=wechat_session).

<span id="ref4">[4]</span> LaTeX工作室. [LaTeX技巧932：如何配置Visual Studio Code作为LaTeX编辑器新版更新](https://www.latexstudio.net/archives/12260.html).
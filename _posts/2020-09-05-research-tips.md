---
layout: post
title:  "科研Tips（文献下载和管理）"
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
* [文献下载](#文献下载)
  * [谷歌学术下载](#谷歌学术下载)
  * [arXiv检索和下载](#arXiv检索和下载)
* [Zotero文献管理](#Zotero文献管理)
* [参考文献](#参考文献)

# 文献下载

## 谷歌学术下载

本方法一般需要翻墙。访问 https://scholar.google.com，搜索文献题目，若该文献有能够下载的来源，右侧会出现包含 `[PDF]` 的下载连接。

![image-20200905151132880](..\assets\img\postsimg\20200905\0.5.jpg)

谷歌学术查询的文献，还可以查询到引用信息，作为额外的信息参考。还可以导出为 bibTex 格式。

![image-20200905202241257](..\assets\img\postsimg\20200905\0.6.jpg)

## arXiv检索和下载

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

最后，测试配置是否成功，下面是 arxiv 上的一篇文章，点击pdf测试下载速度。之后可以手动去掉“cn.”前缀对比速度。 [Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575)

![image-20200905144148277](..\assets\img\postsimg\20200905\1.jpg)

另外，由于 [http://cn.arxiv.org](https://links.jianshu.com/go?to=http%3A%2F%2Fcn.arxiv.org%2F) 并不是主站点，是 arxiv 在中国区的镜像，因此更新有大约半天的延迟，对于当天提交的文章，可能更新不及时。对于当天文章可以手动删除“cn.”前缀解决。 如果出现 pdf 正在自动从源文件生成等提示，为正常现象，稍后即可获取pdf论文。

# Zotero文献管理

文献管理软件可以有效的帮助研究人员管理参考文献，加速论文写作过程。这里介绍开源的文献管理软件 Zotero 的基本功能。**注意**，此处假设主要以英文论文书写为例，若以中文写作为主要工作，可使用 NoteExpress 作为文献管理软件，一般各高校图书馆均提供学校特别版下载安装，NoteExpress 可方便的从中国知网下载和导入文献。

Zotero 官网（https://www.zotero.org/）可以免费下载该软件，软件支持中文。下载安装后，打开「编辑」-「首选项」，切换到「高级」- 「文件和文件夹]选项卡，可以将参考文献索引和存放路径修改到「**自定义的数据存储位置**」：

![image-20200905145730143](..\assets\img\postsimg\20200905\1.5.jpg)

将下载到本地的 pdf 文献拖入 Zotero 软件界面即可添加该文献，等待一会儿后，软件会自动分析出论文的信息并形成一个条目，并将该文献的 pdf 文件拷贝至前面设置的自定义的数据存储位置，因此刚下载到本地的 pdf 文件即可删除了。

![image-20200905144646206](..\assets\img\postsimg\20200905\2.jpg)

点击右侧的第二栏 “笔记” 可以查看和增删对该论文的笔记

![image-20200905144801449](..\assets\img\postsimg\20200905\3.jpg)

双击该条目，可以打开外部 pdf 查看器来查看论文。右键该条目，可以转到 pdf 文件的存放位置（自定义的数据存储位置），或者导出该文献的引文目录。

![image-20200905144924441](..\assets\img\postsimg\20200905\4.jpg)

根据写文章所需要的参考文献格式（此处以 IEEE 为例），选择引文目录，然后选择复制到剪贴板，即可在参考文献中插入自动复制的引文条目：

> [1] H. Hu, J. Gu, Z. Zhang, J. Dai, and Y. Wei, “Relation Networks for Object Detection,” *arXiv:1711.11575 [cs]*, Jun. 2018, Accessed: Sep. 05, 2020. [Online]. Available: http://arxiv.org/abs/1711.11575.

![image-20200905144948669](E:\GitHub\sirlis.github.io\assets\img\postsimg\20200905\5.jpg)

Zotero 另一个优点是可以建立多层级的目录树，便于分门别类的整理参考文献条目。

![image-20200905150036383](..\assets\img\postsimg\20200905\6.jpg)

另一个方法是针对每一篇写的文章建立一个目录，将所有引用的参考文献放入其中，这样可以统一将整个目录的所有参考文献导出参考文献列表。

当注册了Zotero账户后，可以方便的将整个文献库进行云端存储和同步，当两地办公写作时，可以在一处更新参考文献库后，在另一处联网同步，即可无缝保持两地的文献库的同步。账户设置位于「首选项」-「同步」中，设置完毕后，点击软件界面右上角的小图标即可进行同步。

![image-20200905150512124](..\assets\img\postsimg\20200905\7.jpg)

更进一步，可以下载 zotfile 插件（https://www.worldlink.com.cn/en/osdir/zotfile.html），增强 Zotero 的文献管理功能。比如一键规范所有条目的文献 pdf 文件的标题。

# 参考文献

<span id="ref1">[1]</span> [德谟赛斯](https://www.jianshu.com/u/06ba6c212ceb). [如何快速下载 arxiv 论文](https://www.jianshu.com/p/184799230f20).

<span id="ref2">[2]</span>  [strange_jiong](https://blog.csdn.net/dream_allday). [Latex编译出现字体获取不到的情况](https://blog.csdn.net/dream_allday/article/details/84997874).

<span id="ref3">[3]</span>  [开心鲨鱼](https://www.zhihu.com/people/kai-xin-sha-yu). [配置VScode编辑LaTeX及正反向搜索等设置](https://zhuanlan.zhihu.com/p/90526218?utm_source=wechat_session).

<span id="ref4">[4]</span> LaTeX工作室. [LaTeX技巧932：如何配置Visual Studio Code作为LaTeX编辑器新版更新](https://www.latexstudio.net/archives/12260.html).
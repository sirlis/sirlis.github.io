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

最后，测试配置是否成功，下面是 arxiv 上的一篇文章，点击pdf测试下载速度。之后可以手动去掉“cn.”前缀对比速度。 [Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575)

![image-20200905144148277](..\assets\img\postsimg\20200905\1.jpg)

另外，由于 [http://cn.arxiv.org](https://links.jianshu.com/go?to=http%3A%2F%2Fcn.arxiv.org%2F) 并不是主站点，是 arxiv 在中国区的镜像，因此更新有大约半天的延迟，对于当天提交的文章，可能更新不及时。对于当天文章可以手动删除“cn.”前缀解决。 如果出现 pdf 正在自动从源文件生成等提示，为正常现象，稍后即可获取pdf论文。

# Zotero文献管理

文献管理软件可以有效的帮助研究人员管理参考文献，加速论文写作过程。这里介绍开源的文献管理软件 Zotero 的基本功能。

将下载到本地的 pdf 论文拖入 Zotero 软件界面即可添加该文献，等待一会儿后，软件会自动分析出论文的信息并形成一个条目。

![image-20200905144646206](..\assets\img\postsimg\20200905\2.jpg)

点击右侧的第二栏 “笔记” 可以查看和增删对该论文的笔记

![image-20200905144801449](..\assets\img\postsimg\20200905\3.jpg)

双击该条目，可以打开外部 pdf 查看器来查看论文。右键该条目，可以转到 pdf 文件的存放位置，或者导出该文献的引文目录。

![image-20200905144924441](..\assets\img\postsimg\20200905\4.jpg)

根据写文章所需要的参考文献格式（此处以 IEEE 为例），选择引文目录，然后选择复制到剪贴板，即可在参考文献中插入自动复制的引文条目：

> [1] H. Hu, J. Gu, Z. Zhang, J. Dai, and Y. Wei, “Relation Networks for Object Detection,” *arXiv:1711.11575 [cs]*, Jun. 2018, Accessed: Sep. 05, 2020. [Online]. Available: http://arxiv.org/abs/1711.11575.

![image-20200905144948669](E:\GitHub\sirlis.github.io\assets\img\postsimg\20200905\5.jpg)

# 参考文献

<span id="ref1">[1]</span> [德谟赛斯](https://www.jianshu.com/u/06ba6c212ceb). [如何快速下载 arxiv 论文](https://www.jianshu.com/p/184799230f20).

<span id="ref2">[2]</span>  [strange_jiong](https://blog.csdn.net/dream_allday). [Latex编译出现字体获取不到的情况](https://blog.csdn.net/dream_allday/article/details/84997874).

<span id="ref3">[3]</span>  [开心鲨鱼](https://www.zhihu.com/people/kai-xin-sha-yu). [配置VScode编辑LaTeX及正反向搜索等设置](https://zhuanlan.zhihu.com/p/90526218?utm_source=wechat_session).

<span id="ref4">[4]</span> LaTeX工作室. [LaTeX技巧932：如何配置Visual Studio Code作为LaTeX编辑器新版更新](https://www.latexstudio.net/archives/12260.html).
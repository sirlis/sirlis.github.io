---
title: 欢迎使用Jekyll!
date: 2015-03-08 22:21:49 +0800
categories: [Tutorial, Writing]
tags: [html]
math: true
---

本文介绍了基于 jekyll 的个人博客环境的搭建方法，并介绍了部署到 github pages 的方法。

<!--more-->

 ---
 
- [1. 前言](#1-前言)
- [2. 图片](#2-图片)
- [3. 公式](#3-公式)
- [4. 关于页面](#4-关于页面)

# 1. 前言

博客主题位于：https://github.com/cotes2020/jekyll-theme-chirpy

> 注意，本文内容可能随着时间的推移变得过时。
{: .prompt-tip }

# 2. 图片

推荐在 `assets/` 下面新建文件夹存放帖子图片，如 `assets/postimg/`。

在 md 文件中直接使用路径，如 `![img description](/assets/img/postsimg/20200505/1.jpg)` 即可。

帖子的图片无论是在本地（如VSCode打开xxx.github.io文件夹作为根目录）还是在线上（如 `https://xxx.github.io`）均能正常显示。

# 3. 公式

Jekyll支持Markdown的公式。在每个帖子开头增加 `math: true` 即可。他通过调用在线的 mathjax 来渲染。

```yaml
---
title: xxxx
date: 20xx-xx-xx xx:xx:xx +0x00
categories: [xxx, aaa]
tags: [xxa]
math: true # <-- add this
---
```

# 4. 关于页面

自行编辑 `_tabs/about.md` 文件。
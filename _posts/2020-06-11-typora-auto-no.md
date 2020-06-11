---
layout: post
title:  "Typora标题自动编号"
date:   2020-06-11 12:21:49
categories: Tech
tags: Post
---

# 目录

* [目录](#目录)
* [VSCode简介](#VSCode简介)
* [VSCode下载与安装](#VSCode下载与安装)
* [配置C/C\+\+开发环境](#配置C/C\+\+开发环境)
  * [部署编译器](#部署编译器)
  * [生成配置文件](#生成配置文件)
    * [语言配置（c\_cpp\_properties\.json）](#语言配置（c\_cpp\_properties\.json）)
    * [编译配置（task\.json）](#编译配置（task\.json）)
    * [调试配置（launch\.json）](#调试配置（launch\.json）)
  * [编译调试运行](#编译调试运行)
* [参考文献](#参考文献)

# 正文标题自动添加编号

根据官方文档（http://support.typora.io/Auto-Numbering/），首先打开Typora，选择 “文件” => “偏好设置”，切换到 “外观” 选项卡，点击 “打开主题文件夹按钮”

![主题文件夹](..\assets\img\postsimg\20200611\01.preference.png)

然后新建一个 `.txt` 文件，重命名为 `base.user.css` 文件，填充下列代码

```css
/**************************************
 * Header Counters in Content
 **************************************/

/** initialize css counter */
#write {
    counter-reset: h1
}

h1 {
    counter-reset: h2
}

h2 {
    counter-reset: h3
}

h3 {
    counter-reset: h4
}

h4 {
    counter-reset: h5
}

h5 {
    counter-reset: h6
}

/** put counter result into headings */
#write h1:before {
    counter-increment: h1;
    content: counter(h1) ". "
}

#write h2:before {
    counter-increment: h2;
    content: counter(h1) "." counter(h2) ". "
}

#write h3:before,
h3.md-focus.md-heading:before /** override the default style for focused headings */ {
    counter-increment: h3;
    content: counter(h1) "." counter(h2) "." counter(h3) ". "
}

#write h4:before,
h4.md-focus.md-heading:before {
    counter-increment: h4;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) ". "
}

#write h5:before,
h5.md-focus.md-heading:before {
    counter-increment: h5;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) "." counter(h5) ". "
}

#write h6:before,
h6.md-focus.md-heading:before {
    counter-increment: h6;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) "." counter(h5) "." counter(h6) ". "
}

/** override the default style for focused headings */
#write>h3.md-focus:before,
#write>h4.md-focus:before,
#write>h5.md-focus:before,
#write>h6.md-focus:before,
h3.md-focus:before,
h4.md-focus:before,
h5.md-focus:before,
h6.md-focus:before {
    color: inherit;
    border: inherit;
    border-radius: inherit;
    position: inherit;
    left:initial;
    float: none;
    top:initial;
    font-size: inherit;
    padding-left: inherit;
    padding-right: inherit;
    vertical-align: inherit;
    font-weight: inherit;
    line-height: inherit;
}
```

关闭Typora后重启，标题就将会自动增加数字编号。

如果不想要数字与标题之间的 `.` ，则需要修改 `.css` 文件中，`/** put counter result into headings */` 后面的代码，将每个标题最后一个 `". "` 中的句点去掉，即将

```css
/** put counter result into headings */
#write h1:before {
    counter-increment: h1;
    content: counter(h1) ". " /* <--- delete this dot */
}
...
```

改为

```css
/** put counter result into headings */
#write h1:before {
    counter-increment: h1;
    content: counter(h1) " " /* <--- dot deleted */
}
...
```

# 目录自动编号

如果想要在自动生成的目录中附带编号，需要额外再在 `.css` 文件中增加代码段[[2](#ref2)]

```css
/**************************************
 * Header Counters in TOC
 **************************************/
 
/* No link underlines in TOC */
.md-toc-inner {
    text-decoration: none;
}
 
.md-toc-content {
    counter-reset: h1toc
}
 
.md-toc-h1 {
    margin-left: 0;
    font-size: 1.5rem;
    counter-reset: h2toc
}
 
.md-toc-h2 {
    font-size: 1.1rem;
    margin-left: 2rem;
    counter-reset: h3toc
}
 
.md-toc-h3 {
    margin-left: 3rem;
    font-size: .9rem;
    counter-reset: h4toc
}
 
.md-toc-h4 {
    margin-left: 4rem;
    font-size: .85rem;
    counter-reset: h5toc
}
 
.md-toc-h5 {
    margin-left: 5rem;
    font-size: .8rem;
    counter-reset: h6toc
}
 
.md-toc-h6 {
    margin-left: 6rem;
    font-size: .75rem;
}
 
.md-toc-h1:before {
    color: black;
    counter-increment: h1toc;
    content: counter(h1toc) ". " /* <--- delete this dot if not wanted */
}
 
.md-toc-h1 .md-toc-inner {
    margin-left: 0;
}
 
.md-toc-h2:before {
    color: black;
    counter-increment: h2toc;
    content: counter(h1toc) ". " counter(h2toc) ". " /* <--- delete this dot if not wanted */
}
 
.md-toc-h2 .md-toc-inner {
    margin-left: 0;
}
 
.md-toc-h3:before {
    color: black;
    counter-increment: h3toc;
    content: counter(h1toc) ". " counter(h2toc) ". " counter(h3toc) ". " /* <--- delete this dot if not wanted */
}
 
.md-toc-h3 .md-toc-inner {
    margin-left: 0;
}
 
.md-toc-h4:before {
    color: black;
    counter-increment: h4toc;
    content: counter(h1toc) ". " counter(h2toc) ". " counter(h3toc) ". " counter(h4toc) ". " /* <--- delete this dot if not wanted */
}
 
.md-toc-h4 .md-toc-inner {
    margin-left: 0;
}
 
.md-toc-h5:before {
    color: black;
    counter-increment: h5toc;
    content: counter(h1toc) ". " counter(h2toc) ". " counter(h3toc) ". " counter(h4toc) ". " counter(h5toc) ". " /* <--- delete this dot if not wanted */
}
 
.md-toc-h5 .md-toc-inner {
    margin-left: 0;
}
 
.md-toc-h6:before {
    color: black;
    counter-increment: h6toc;
    content: counter(h1toc) ". " counter(h2toc) ". " counter(h3toc) ". " counter(h4toc) ". " counter(h5toc) ". " counter(h6toc) ". " /* <--- delete this dot if not wanted */
}
 
.md-toc-h6 .md-toc-inner {
    margin-left: 0;
}
```

若要移除数字与标题间的句点，类似上边的做法，将 `". "` 改为 `" "`。

# 侧边栏自动编号

如果想要在Typora的侧边栏中附带编号，需要额外再在 `.css` 文件中继续增加代码段[[3](#ref3)]

```css
/**************************************
 * Header Counters in Sidebar
 **************************************/

.sidebar-content {
    counter-reset: h1
}
 
.outline-h1 {
    counter-reset: h2
}
 
.outline-h2 {
    counter-reset: h3
}
 
.outline-h3 {
    counter-reset: h4
}
 
.outline-h4 {
    counter-reset: h5
}
 
.outline-h5 {
    counter-reset: h6
}
 
.outline-h1>.outline-item>.outline-label:before {
    counter-increment: h1;
    content: counter(h1) ". " /* <--- delete this dot if not wanted */
}
 
.outline-h2>.outline-item>.outline-label:before {
    counter-increment: h2;
    content: counter(h1) "." counter(h2) ". " /* <--- delete this dot if not wanted */
}
 
.outline-h3>.outline-item>.outline-label:before {
    counter-increment: h3;
    content: counter(h1) "." counter(h2) "." counter(h3) ". " /* <--- delete this dot if not wanted */
}
 
.outline-h4>.outline-item>.outline-label:before {
    counter-increment: h4;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) ". " /* <--- delete this dot if not wanted */
}
 
.outline-h5>.outline-item>.outline-label:before {
    counter-increment: h5;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) "." counter(h5) ". " /* <--- delete this dot if not wanted */
}
 
.outline-h6>.outline-item>.outline-label:before {
    counter-increment: h6;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) "." counter(h5) "." counter(h6) ". " /* <--- delete this dot if not wanted */
}
```

若要移除数字与标题间的句点，类似上边的做法，将 `". "` 改为 `" "`。

# 参考文献

<span id="ref1">[1]</span> Typora. [Auto Numbering for Headings](http://support.typora.io/Auto-Numbering/).

<span id="ref2">[2]</span> Guest. [TOC Autonumbering for Typora](https://pastebin.com/NYugSbXk).

<span id="ref3">[3]</span> Guest. [Auto numbering Typora outline](https://pastebin.com/XmYgBbaz).
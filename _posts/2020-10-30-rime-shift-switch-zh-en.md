---
title: Typora标题自动编号
date: 2020-10-30 22:06:49 +0800
categories: [Tutorial, Writing]
tags: [markdown]
math: true
---

本文记录个人科研生活中的各种小tips和遇到的问题及其解决方案，留作提醒查阅。

<!--more-->

 ---

- [1. Rime 输入法](#1-rime-输入法)
  - [1.1. shift 直接上屏且切换中英文](#11-shift-直接上屏且切换中英文)
- [2. 参考文献](#2-参考文献)

# 1. Rime 输入法

## 1.1. shift 直接上屏且切换中英文

Xeon-Shao. [小狼毫（Rime）输入法设置Shift直接上屏英文字符并切换为英文状态方法](https://blog.csdn.net/sdujava2011/article/details/84098971)

小狼毫默认输入方式下，左Shift键只切换为英文，右Shift键直接上屏中文。这对于用惯了搜狗的人来说在进行中英文混输的时候经常出错，特别影响效率，接下来提供方法解决这个问题。 

- 开始菜单中，找到小狼毫的文件夹，打开 “**用户资料文件夹**”，如果有 `weasel.custom.yaml`，则复制一份，复制后的文件重命名为 `default.custom.yaml`；如果没有，则自己新建文件命名为 `default.custom.yaml`，并将内容修改为如下：

```
customization:
  distribution_code_name: Weasel
  distribution_version: 0.12.0
  generator: "Weasel::UIStyleSettings"
  modified_time: "Thu Nov 15 09:43:07 2018"
  rime_version: 1.3.2
patch:
  "ascii_composer/switch_key/Shift_L": commit_code
```

- 小狼毫文件夹里点击 “**小狼毫重新部署**”；

`Shift_L: commit_code` 的意思是：其中 `L` 指 `Left`，左的意思。`commit-` 提交。`code－` 代码。 当我们输入一段文字未上屏之前，按此键后字符将被将直接上屏。

# 2. 参考文献

无。
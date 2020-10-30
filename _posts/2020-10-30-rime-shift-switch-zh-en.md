---
title: 日常tips手册
date: 2020-10-30 22:06:49 +0800
categories: [Diary]
tags: [efficiency]
math: true
---

本文记录个人科研生活中的各种小tips和遇到的问题及其解决方案，留作提醒查阅。

<!--more-->

 ---

- [1. Rime 输入法](#1-rime-输入法)
  - [基础配置](#基础配置)
  - [1.1. shift 直接上屏且切换中英文](#11-shift-直接上屏且切换中英文)
- [2. 参考文献](#2-参考文献)

# 1. Rime 输入法

## 基础配置

XNOM. [30分钟搞定 自由输入法RIME简明配置指南](https://www.jianshu.com/p/296bba666604)

- Rime 的各种配置，均是由 `.yaml` 文件所定义。`yaml` 是一种标记语言。`.yaml` 文件实际上是文本文档。可使用记事本、或 Emeditor 等进行编辑。

- 对 Rime 进行自定义，是通过对 `.custom.yaml` 文件修改达成。不同的 `.custom.yaml` 文件，控制不同的功能实现。`.custom.yaml` 实际上是相当于对 `.yaml` 文件打补丁，在重新部署后，会将 `.custom.yaml` 中的内容写入 `.yaml` 文件中，完成自定。

    - 例一：`weasel.yaml` 是常规设置，主要控制托盘图标、候选词横竖排列、界面配色等等功能。那么，我们需要定制界面配色，只需在 `weasel.custom.yaml` 中修改，重新部署后就可实现。
    - 例二：`default.yaml` 是默认设置，主要控制快捷键、按键上屏等等。同样，作修改就编辑 `default.custom.yaml` 文件即可。
    - 例三：以上是全局设置，亦即不论使用何种输入方案，均起作用。`double_pinyin_flypy.custom.yaml` 这种则是输入法方案设置。主要实现特殊标点符号、词库等功能。是针对特定输入方案的配置。

可见，我们绝大部分的自定，都只需修改对应的 `.custom.yaml` 文件即可。

- 所有自定修改，都必须重新部署。在开始菜单可以找到【小狼毫】重新部署。

作者：XNOM
链接：https://www.jianshu.com/p/296bba666604
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

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
---
title: 日常tips手册（Rime输入法）
date: 2020-10-30 22:06:49 +0800
categories: [Diary]
tags: [efficiency]
math: true
---

本文记录个人科研生活中的各种小tips和遇到的问题及其解决方案，留作提醒查阅。

<!--more-->

 ---

- [1. Rime 输入法](#1-rime-输入法)
  - [1.1. 基础配置](#11-基础配置)
  - [1.2. 皮肤设置](#12-皮肤设置)
  - [1.3. 特殊符号快速输入](#13-特殊符号快速输入)
  - [1.4. shift 直接上屏且切换中英文](#14-shift-直接上屏且切换中英文)
- [2. 参考文献](#2-参考文献)

# 1. Rime 输入法

## 1.1. 基础配置

XNOM. [30分钟搞定 自由输入法RIME简明配置指南](https://www.jianshu.com/p/296bba666604)

- Rime 的各种配置，均是由 `.yaml` 文件所定义。`yaml` 是一种标记语言。`.yaml` 文件实际上是文本文档。可使用记事本、或 Emeditor 等进行编辑。

- 对 Rime 进行自定义，是通过对 `.custom.yaml` 文件修改达成。不同的 `.custom.yaml` 文件，控制不同的功能实现。`.custom.yaml` 实际上是相当于对 `.yaml` 文件打补丁，在重新部署后，会将 `.custom.yaml` 中的内容写入 `.yaml` 文件中，完成自定。

    - 例一：`weasel.yaml` 是常规设置，主要控制托盘图标、候选词横竖排列、界面配色等等功能。那么，我们需要定制界面配色，只需在 `weasel.custom.yaml` 中修改，重新部署后就可实现。
    - 例二：`default.yaml` 是默认设置，主要控制快捷键、按键上屏等等。同样，作修改就编辑 `default.custom.yaml` 文件即可。
    - 例三：以上是全局设置，亦即不论使用何种输入方案，均起作用。`double_pinyin_flypy.custom.yaml` 这种则是输入法方案设置。主要实现特殊标点符号、词库等功能。是针对特定输入方案的配置。

可见，我们绝大部分的自定，都只需修改对应的 `.custom.yaml` 文件即可。

- 所有自定修改，都必须重新部署。在开始菜单可以找到【小狼毫】重新部署。

## 1.2. 皮肤设置

XNOM. [30分钟搞定 自由输入法RIME简明配置指南](https://www.jianshu.com/p/296bba666604)

打开 `weasel.custom.yaml` 文件，若没有，则新建。所有自定义项均在 `patch:` 下，注意缩进

```yaml
customization:
  distribution_code_name: Weasel
  distribution_version: 0.14.3
  generator: "Weasel::UIStyleSettings"
  modified_time: "Mon Jul 13 11:31:05 2020"
  rime_version: 1.5.3
patch:
  "style/color_scheme": google # 皮肤风格
  "style/layout/border_width": 0
  "style/layout/border": 0
  "style/horizontal": true #横排显示候选词
  "style/font_face": Microsoft YaHei # 候选词字体
  "style/font_point": 12 # 候选词字号
```

一个模仿 Windows 10 自带的微软拼音皮肤的设置如下，以供修改尝试参考：

```yaml
customization:
  distribution_code_name: Weasel
  distribution_version: 0.14.3
  generator: "Weasel::UIStyleSettings"
  modified_time: "Thu Jun 27 17:32:21 2019"
  rime_version: 1.5.3

patch:
  "style/display_tray_icon": true
  "style/horizontal": true #横排显示
  "style/font_face": "Microsoft YaHei" #字体
  "style/font_point": 13 #字体大小
  "style/inline_preedit": true # 嵌入式候选窗单行显示

  "style/layout/border_width": 0
  "style/layout/border": 0
  "style/layout/margin_x": 12 #候选字左右边距
  "style/layout/margin_y": 12 #候选字上下边距
  "style/layout/hilite_padding": 12 #候选字背景色色块高度 若想候选字背景色块无边界填充候选框，仅需其高度和候选字上下边距一致即可
  "style/layout/hilite_spacing": 3 # 序号和候选字之间的间隔
  "style/layout/spacing": 10 #作用不明
  "style/layout/candidate_spacing": 24 # 候选字间隔
  "style/layout/round_corner": 0 #候选字背景色块圆角幅度

  "style/color_scheme": Micosoft
  "preset_color_schemes/Micosoft":
    name: "Micosoft"
    author: "XNOM"
    back_color: 0xffffff #候选框 背景色
    border_color: 0xD77800 #候选框 边框颜色
    text_color: 0x000000 #已选择字 文字颜色
    hilited_text_color: 0x000000 #已选择字右侧拼音 文字颜色
    hilited_back_color: 0xffffff #已选择字右侧拼音 背景色
    hilited_candidate_text_color: 0xffffff #候选字颜色
    hilited_candidate_back_color: 0xD77800 #候选字背景色
    candidate_text_color: 0x000000 #未候选字颜色
```

## 1.3. 特殊符号快速输入

百度贴吧. [小狼毫输入法怎么输入希腊字母和数学符号](https://tieba.baidu.com/p/3079474120)

首先确定使用的拼音方案，比如如果使用 `luna_pinyin_simp` （明月拼音简化字）方案，那么新建 `lunar_pinyin_simp.custom.yaml` 配置文件，写入：

```yaml
patch:
  "punctuator/import_preset" : symbols
  "recognizer/patterns/punct": "^/([A-Z|a-z]*|[0-9]|10)$"
```

然后 **小狼毫重新部署**，即可。

通过 `/` 键配合缩写实现快速输入。注意使用键盘的 `?` 键唤起，而不是小键盘的 `/` 键。

- 输入 `/xl` 直接给出希腊字母，如 $\alpha$ 等。
- 输入 `/sx` 直接给出常用数学符号，如 ±, ÷ 等。

## 1.4. shift 直接上屏且切换中英文

Xeon-Shao. [小狼毫（Rime）输入法设置Shift直接上屏英文字符并切换为英文状态方法](https://blog.csdn.net/sdujava2011/article/details/84098971)

小狼毫默认输入方式下，左Shift键只切换为英文，右Shift键直接上屏中文。这对于用惯了搜狗的人来说在进行中英文混输的时候经常出错，特别影响效率，接下来提供方法解决这个问题。 

- 开始菜单中，找到小狼毫的文件夹，打开 “**用户资料文件夹**”，如果有 `weasel.custom.yaml`，则复制一份，复制后的文件重命名为 `default.custom.yaml`；如果没有，则自己新建文件命名为 `default.custom.yaml`，并将内容修改为如下：

```yaml
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
---
layout: post
title:  "Windows10通过应用商店安装Ubuntu"
date:   2020-07-26 10:55:19
categories: Coding
tags: Linux
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
* [环境配置](#环境配置)
* [下载和安装](#下载和安装)
* [注意事项](#注意事项)
* [参考文献](#参考文献)

# 环境配置

必须先启用“适用于 Linux 的 Windows 子系统”可选功能，然后才能在 Windows 上安装 Linux 分发版。

以管理员身份打开 PowerShell 并运行

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

或者通过控制面板进行界面配置（需要重启）

![](..\assets\img\postsimg\20200726\1.jpg)

# 下载和安装

打开 [Microsoft Store](https://aka.ms/wslstore)，并选择你偏好的 Linux 分发版。

选择好版本后，选择「获取」，然后点击「安装」。如果多次点击并无反应，可以等待一定时间后返回到Microsoft Store首页，再次前往该分发版的 Linux 应用页面，该应用可能已经在后台安装完毕。

![](..\assets\img\postsimg\20200726\2.jpg)

![](..\assets\img\postsimg\20200726\3.jpg)

点击「启动」，将打开一个控制台窗口，系统会要求你等待一分钟或两分钟，以便文件解压缩并存储到电脑上。 未来的所有启动时间应不到一秒。然后，需要[为新的 Linux 分发版创建用户帐户和密码](https://docs.microsoft.com/zh-cn/windows/wsl/user-support)。（注意，linux 系统中，输入密码时默认时不显示任何内容的）

![](..\assets\img\postsimg\20200726\4.jpg)

提示 `Installation successful!` 表明系统安装完毕。

通过 Windows Store 安装的 Linux 子系统，其存放路径位于

```powershell
C:\Users\[YourUserName]\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04onWindows_79rhkp1fndgsc\LocalState\rootfs
```

其中 `[YourUserName]` 是个人的电脑用户名，`CanonicalGroupLimited.XXX` 是相应的子系统版本。通过上述路径可以方便的进行文件管理。

# 注意事项

- **WslRegisterDistribution 失败并出现错误 0x8007019e**
  - 未启用“适用于 Linux 的 Windows 子系统”可选组件：
  - 打开“控制面板” -> “程序和功能” -> “打开或关闭 Windows 功能”-> 选中“适用于 Linux 的 Windows 子系统”，或使用本文开头所述的 PowerShell cmdlet。
- **安装失败，出现错误 0x80070003 或错误 0x80370102**
  - 请确保在计算机的 BIOS 内已启用虚拟化。 有关如何执行此操作的说明因计算机而异，并且很可能在 CPU 相关选项下。
- **尝试升级时出错：`Invalid command line option: wsl --set-version Ubuntu 2`**
  - 请确保已启用适用于 Linux 的 Windows 子系统，并且你使用的是 Windows 内部版本 19041 或更高版本。 若要启用 WSL，请在 PowerShell 提示符下以具有管理员权限的身份运行此命令：`Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux`。 可在[此处](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10)找到完整的 WSL 安装说明。
- **由于虚拟磁盘系统的某个限制，无法完成所请求的操作。虚拟硬盘文件必须是解压缩的且未加密的，并且不能是稀疏的。**
  - 请检查 [WSL GitHub 主题 #4103](https://github.com/microsoft/WSL/issues/4103)，其中跟踪了此问题以提供更新的信息。
- **无法将词语“wsl”识别为 cmdlet、函数、脚本文件或可运行程序的名称。**
  - 请确保[已安装“适用于 Linux 的 Windows 子系统”可选组件](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10#enable-the-virtual-machine-platform-optional-component)。 此外，如果你使用的是 ARM64 设备，并从 PowerShell 运行此命令，则会收到此错误。 请改为从 [PowerShell Core](https://docs.microsoft.com/zh-cn/powershell/scripting/install/installing-powershell-core-on-windows?view=powershell-6) 或从命令提示符运行 `wsl.exe`。

# 参考文献

<span id="ref1">[1]</span>  Microsoft. [适用于 Linux 的 Windows 子系统安装指南 (Windows 10)](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10#install-your-linux-distribution-of-choice).


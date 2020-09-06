---
title: Windows10通过应用商店安装Ubuntu"
date: 2020-07-26 10:55:19 +0800
categories: [Coding]
tags: [linux]
math: true
---

# 环境配置

首先，**启用开发者模式**：菜单栏打开**设置**——点击**更新和安全**——**启用开发人员模式**（时间会有点长）

其次，**更改系统功能**：必须先启用“适用于 Linux 的 Windows 子系统”可选功能，然后才能在 Windows 上安装 Linux 分发版。

以管理员身份打开 PowerShell 并运行

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

或者通过控制面板进行界面配置（需要重启）

![](\assets\img\postsimg\20200726\1.jpg)

# 下载和安装

打开 [Microsoft Store](https://aka.ms/wslstore)，并选择你偏好的 Linux 分发版。

选择好版本后，选择「获取」，然后点击「安装」。如果多次点击并无反应，可以等待一定时间后返回到Microsoft Store首页，再次前往该分发版的 Linux 应用页面，该应用可能已经在后台安装完毕。

![](\assets\img\postsimg\20200726\2.jpg)

![](\assets\img\postsimg\20200726\3.jpg)

点击「启动」，将打开一个控制台窗口，系统会要求你等待一分钟或两分钟，以便文件解压缩并存储到电脑上。 未来的所有启动时间应不到一秒。然后，需要[为新的 Linux 分发版创建用户帐户和密码](https://docs.microsoft.com/zh-cn/windows/wsl/user-support)。（注意，linux 系统中，输入密码时默认时不显示任何内容的）

![](\assets\img\postsimg\20200726\4.jpg)

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

# 基本操作

对 linux 子系统的操作可基于 [windows subsystem for linux (*wsl*)](http://www.baidu.com/link?url=jRq5GQOKupZSX7p973mR5YQ0WwqNWa6Jupvwyo8OR5fHoLw3z_xTeI5O5eoguWLL) 来进行。

Windows键+R，输入 `cmd` 回车，打开命令行窗口。输入`wsl -l`,可以看到我系统里装的 linux 系统发行版本。输入`wsl --version` 可以看到版本信息和命令行参数一览。

## 备份、删除和还原

备份子系统非常简单，但一定要先停止子系统之后再备份

```shell
wsl --export Ubuntu-20.04 c:\XXX\Ubuntu-18.04-20200726.tar
```

待完成即可。备份成功后，子系统会被打包成命令中指定名称的tar文件。

删除子系统也是一个命令即可：

```shell
wsl --unregister Ubuntu-12.04
```

还原子系统。 删除了没关系，刚才做了备份，也是一个命令还原：

```shell
wsl --import Ubuntu-20.04 c:\AAA c:\XXX\Ubuntu-20.04-20200726.tar
```

这里注意指定还原的路径。成功后，子系统又回来了，可以用`wsl -l`确认一下。

## 切换数据源

在 Ubuntu 下我们可以通过 `apt-get` 命令 很方便的安装 / 卸载软件，切换数据源为国内的镜像站点速度会变快。编辑数据源配置文件

```bash
vi /etc/apt/sources.list
```

继续按enter键进入真正的vi编辑页面

> 科普：
> vi编辑器一共有三种模式： 命令模式（command mode） 插入模式（Insert mode） 底行模式（last line mode） 命令模式下我们只能控制屏幕光标的移动，字符、字或行的删除，移动复制某区段及进入Insert mode下，或者到 last line mode等；插入模式下可以做文字输入，按「ESC」键可回到命令行模式；底行模式下，可以将文件保存或退出vi，也可以设置编辑环境，如寻找字符串、列出行号等。

当我们进入vi编辑器的时候默认是命令行模式，这时如果想编辑内容，就输入 i 命令就可以了。现在我们要把镜像源改为阿里的，所以插入如下内容：

```bash
deb http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse
```


接着按「ESC」退会命令行模式，输入命令：

```
: wq!
```

保存退出就好了。接着输入命令

```bash
sudo apt-get update
```

然后输入密码，回车，更新配置就可以了，飞速！

如果命令行修改文件不习惯（linux下就是这样），可在windows下直接找到文件，用记事本打开后修改保存。文件路径在比如

```
C:\Users\[YourUserName]\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04onWindows_79rhkp1fndgsc\LocalState\rootfs\etc\apt
```

## Windows访问Linux文件

通过Windows Store 安装的 Linux 子系统位于

```
C:\Users\[YourUserName]\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04onWindows_79rhkp1fndgsc\LocalState\rootfs
```

也可以运行子系统后，输入

```
explorer.exe .
```

在资源管理器种打开相应的文件夹。

## Linux访问Windows文件

在bash种输入以下命令，即为windows系统下

```
cd /mnt
```

接着就是不断的cd进入到你所需的目录下。比如我们进入系统后 `dir` 或者 `ls` 一下就可以看到目前 Windows 系统的三个盘符（因人而异）。`cd` 就可以逐级进入文件夹。

![image-20200726135034927](\assets\img\postsimg\20200726\5.jpg)

## 解压tar文件

在 Linux 系统种，通过 `tar` 命令解压 tar 压缩包。

![image-20200726141328495](\assets\img\postsimg\20200726\6.jpg)



# 参考文献

<span id="ref1">[1]</span>  Microsoft. [适用于 Linux 的 Windows 子系统安装指南 (Windows 10)](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10#install-your-linux-distribution-of-choice).


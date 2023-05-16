---
title: Windows环境下使用MinGW-W64、CMake和NSIS打包C++工程
date: 2023-05-15 23:59:19 +0800
categories: [Tutorial, Coding]
tags: [vscode, c/c++, cmake, cpack, mingw64]
math: true
---

本文介绍了 Windows 环境下使用 CMake（CPack） 和 NSIS 构建并打包 C/C++ 工程项目的基本流程和方法，核心在于 CMakeLists.txt 文件的编写。

<!--more-->

---

- [1. 引言](#1-引言)
- [CMake 配置](#cmake-配置)
  - [环境搭建](#环境搭建)
  - [配置（Configure）](#配置configure)
  - [生成（Generate）](#生成generate)
  - [构建（Build）](#构建build)
  - [运行和调试](#运行和调试)
- [基于 CMake 的打包](#基于-cmake-的打包)
  - [CPack](#cpack)
  - [NSIS](#nsis)
  - [基于 CPack 和 NSIS 的打包](#基于-cpack-和-nsis-的打包)
- [参考文献](#参考文献)


## 1. 引言  


前面我们介绍了在VSCode中采用MinGW-W64作为C/C++的编译器来进行C++编程。为了进一步对构建完成的工程项目进行打包分发，需要采用CMake和NSIS来对项目进行编译打包。


## CMake 配置

CMake（官网：https://cmake.org）是一个跨平台的安装（编译）工具，可以用简单的语句来描述所有平台的安装(编译过程)。他能够输出各种各样的makefile或者project文件，能测试编译器所支持的C++特性

你或许听过好几种 Make 工具，例如 GNU Make ，QT 的 qmake ，微软的 MS nmake，BSD Make（pmake），Makepp，等等。这些 Make 工具遵循着不同的规范和标准，所执行的 Makefile 格式也千差万别。这样就带来了一个严峻的问题：如果软件想跨平台，必须要保证能够在不同平台编译。而如果使用上面的 Make 工具，就得为每一种标准写一次 `Makefile` ，这将是一件让人抓狂的工作。CMake 就是针对上面问题所设计的工具：它首先允许开发者编写一种平台无关的 `CMakeLists.txt` 文件来定制整个编译流程，然后再根据目标用户的平台进一步生成所需的本地化 Makefile 和工程文件，如 Unix 的 Makefile 或 Windows 的 Visual Studio 工程。从而做到“Write once, run everywhere”。

> 比如，同一个工程源码，可以通过 CMake 得到 Visual Studio 系列软件使用的 `.sln` 和 `.vcproj` 工程文件，也可以生成 `makefile` 文件。

显然，CMake 是一个比上述几种 make 更高级的编译配置工具。一些使用 CMake 作为项目架构系统的知名开源项目有 VTK、ITK、KDE、OpenCV、OSG 等

### 环境搭建

前提，已经安装有至少一个C/C++编译器，如 MinGW-W64。可以通过`gcc --version` 查看版本来确认安装是否成功。

在 Windows 环境下，直接前往官网下载最新的安装包安装。可以通过 `cmake --version` 查看版本来确认安装是否成功。

在 VScode 中，需要安装以下两个插件：
- CMake
- CMake Tools

使用 CMake 插件创建 `CMakeLists.txt` 文件（文件名一个字都不能错）。两种创建方式，推荐后者：

- 手动创建，直接在工程项目的根目录下新建一个 `CMakeLists.txt` 文件；

- 【推荐】插件自动创建，在 VSCode 中打开工程项目文件夹，输入快捷键组合 `Ctrl + Shift + P` 然后输入 `cmake quick start` 进行快速设置。首次设置会弹出 `Select a Kit` 需要选择一个编译器，若正确安装 MinGW-W64 并添加了环境变量，一般会自动检索到类似 `GCC XX.X.X x86-64-w64-mingw32` 的编译器，注意检查后面的路径是否正确，然后选择即可。选择后即会在项目根目录下自动创建`CMakeLists.txt` 文件。

![](/assets/img/postsimg/20230515/select-a-kit.jpg)

### 配置（Configure）

配置即编写 `CMakeLists.txt` 文件，通常一个 `CMakeLists.txt` 文件需要包含：

- `project(xxx)`，必须
- `add_subdirectory(子文件夹名称)`，若父目录包含多个子目录则必须
- `add_library(库文件名称 STATIC 文件)`，通常子目录(二选一)
- `add_executable(可执行文件名称 文件)`，通常父目录(二选一)
- `include_directories(路径)`，必须
- `link_directories(路径)`，非必须
- `target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)`，必须

一个典型的 `CMakeLists.txt` 文件如下：

```cmake
cmake_minimum_required(VERSION 3.0.0) # 设置最小的cmake版本号
set(PROJECT_NAME "myproject") # 设置工程名称
project(${PROJECT_NAME} VERSION 0.2.0) # 设置工程版本

set(CMAKE_C_STANDARD 99) # 设置C标准
set(CMAKE_CXX_STANDARD 11) # 设置C++标准

include(CTest)
enable_testing()

set(CMAKE_VERBOSE_MAKEFILE "ON") # 启用详细打印，方便查看bug

### 通过遍历来搜寻所有 .h 文件，并保存到 ALL_H 变量中
# ${PROJECT_SOURCE_DIR} 是默认的工程根目录
file(GLOB_RECURSE ALL_H
        ${PROJECT_SOURCE_DIR}/src/**.h
)

### 遍历每个 .h 文件（这段抄的网上的）
foreach(file  ${ALL_H})
    # 找到文件的上级路径
    string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${file})
    string(REGEX REPLACE "(.*/)(.*)" "\\1" CURRENT_FOLDER ${CURRENT_FOLDER_ABSOLUTE})
    list(APPEND include_h  ${CURRENT_FOLDER})
endforeach()
### 删除冗余路径
list(REMOVE_DUPLICATES  include_h) # 得到去冗的所有包含 .h 文件的路径，存到 include_h 变量
# MESSAGE("include_directories"  ${include_h}) # 打印所有包含 .h 文件的路径

# 规定 .h 文件的路径
include_directories(${include_h})

### 遍历所有 .c/.cpp 文件，存到 ALL_SRC 变量中
file(GLOB_RECURSE ALL_SRC
    ${PROJECT_SOURCE_DIR}/src/**.c
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    )
# MESSAGE("add_executable"  ${ALL_SRC}) # 打印

# 把所有.c/.cpp 文件添加到名字为 ${PROJECT_NAME} 的可执行程序中
add_executable(${PROJECT_NAME} ${ALL_SRC})

### 定义宏定义来控制工程中的一些功能
# 比如代码里通过 #ifdef AASSAA aaa #else bbb
# 那么就可以通过 `-DAASSAA` 传给编译器进行宏定义
# 等价于代码中写 `#define DAASSAA`
add_definitions(-D__INFO__)
add_definitions(-D__CLEAN__)

### 如果是Windows环境
if(WIN32)
    # 对 add_library 或 add_executable 生成的文件进行链接操作
    # 这里额外链接 TCP 通讯所需的 winsock2.h 依赖的 ws2_32.lib
    target_link_libraries(${PROJECT_NAME} ws2_32)

    ### 安装（linux中的sudo apt install 的概念，win中不知道是啥）
    set(CMAKE_PREFIX_PATH "E:/ProgramFiles/Git/mingw64") # 设置编译器路径，用来找程序依赖的额外dll
    # 注意，这里最好使用Depends工具来查看编译得到的可执行程序（.exe）依赖哪些第三方dll，然后逐一添加
    install(FILES "${CMAKE_PREFIX_PATH}/bin/libstdc++-6.dll" DESTINATION .)
    install(FILES "${CMAKE_PREFIX_PATH}/bin/libgcc_s_seh-1.dll" DESTINATION .)
    install(FILES "${CMAKE_PREFIX_PATH}/bin/libwinpthread-1.dll" DESTINATION .)
    # 将可执行程序打包到（将来安装位置的）根目录
    install(TARGETS ${PROJECT_NAME} DESTINATION .)
    # 将一些额外的资源文件夹，配置文件（夹）等拷贝到目标目录
    # 文件夹用 'DIRECTORY' ， 文件用 'FILES'
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/data/ DESTINATION data)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/config/ DESTINATION config)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/res/ DESTINATION res)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".") # 不知道有没有用，先放着了
    include(InstallRequiredSystemLibraries) # 不知道有没有用，应该有用，把系统dll打包到exe

    ### 打包
    # set(CPACK_INSTALL_PREFIX "/home/DSS") # 给linux用的，大概把？
    # 设置一些名字
    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
    set(CPACK_CMAKE_GENERATOR "MinGW Makefiles") # 如果前面用了自动生成CMakeLists.txt，这里也可以不写
    set (CPACK_RESOURCE_FILE_LICENSE  
        "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE") # 如果工程没有License文件也可以不写
    set(CPACK_PACKAGE_VERSION_MAJOR "${${PROJECT_NAME}_VERSION_MAJOR}") # 大版本号
    set(CPACK_PACKAGE_VERSION_MINOR "${${PROJECT_NAME}_VERSION_MINOR}") # 小版本号
    set(CPACK_SOURCE_GENERATOR "TGZ") # 压缩方式，我随便写了一个，支持很多种

    ### 在 Windows 环境种，采用开源的 NSIS 来构建安装程序
    set(CPACK_GENERATOR NSIS)
    set(CPACK_NSIS_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_NSIS_DISPLAY_NAME "${PROJECT_NAME}")

    # ico我还没设置成功...会报错，先注释掉了
    # set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/DSSimulator.svg")
    # set(CPACK_NSIS_MUI_ICON "DSSimulator.svg")

    ### 用来告诉安装程序，卸载的时候需要额外删掉前面 install 时额外加入的 资源文件（夹）和 dll 等
    # 这里采用函数的形式（百度抄的），也可以在外部编写 '.nsi'文件然后引用进来（听着就麻烦）
    function(add_uninstall_command)
        foreach(file IN LISTS ARGN) # ARGN 是参数
            if(IS_DIRECTORY "${file}")
                set(command "rmdir /s /q \"$INSTDIR\\\\${file}\"")
            else()
                set(command "del /f /q \"$INSTDIR\\\\${file}\"")
            endif()
            # 这句就是给 NSIS 提供的删除命令
            set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "${CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS}\n !system '${command}'")
        endforeach()
    endfunction()
    # 调用函数把前面添加的东西删掉
    add_uninstall_command("$INSTDIR/data" "$INSTDIR/config" "$INSTDIR/res" "$INSTDIR/libstdc++-6.dll" "$INSTDIR/libgcc_s_seh.dll" "$INSTDIR/libwinpthread-1.dll")
    # 这句话必须要有哦
    include(CPack)

endif()

### 这是用来提示自己的命令行的命令，不然老年痴呆记不住
# cmake --build . --target install --verbose
# cpack.exe .\CPackConfig.cmake
```

在 `CMakeLists.txt` 文件中：
- 采用 `set(变量 文件名/路径/...)` 函数给文件名/路径名或其他字符串起别名，用 `${变量}` 获取变量内容；
- 采用 `${xxx}` 来取变量的值；
- 采用 `aux_source_directory(路径 变量)` 可获取路径下所有的.cpp/.c/.cc文件（不包括子目录），并赋值给变量;
- 采用 `add_compile_definitions(xxx=1)` 可以给宏具体值，但是只有高版本的cmake支持，等价于 `#define xxx 1`；
- 采用 `add_subdirectory(子文件夹名称)` 编译子文件夹的 `CMakeLists.txt`；
- 如果需要将工程编译为静态库，那么使用 `add_library(库文件名称 STATIC 文件)`。注意，库文件名称通常为 `libxxx.so`，在这里要去掉前后缀写 `xxx` 即可；
- 规定 `.so/.a` 库文件路径使用 `link_directories(路径)`；


### 生成（Generate）

我们一般会在 `CMakeLists.txt` 所在的目录下（一般也就是工程项目的根目录）手动新建一个 `build` 文件夹，这将用于存储 CMake 构建的中间文件和生成的目标文件。这种方式实际上是 cmake 的 out-of-source 构建方式。这种方法可以保证生成中间产物与源代码分离（即生成的所有目标文件都存储在 `build` 文件夹中，因此不会干扰源代码中的任何文件）。

采用命令行的方式可操作如下

```
mkdir build
cd build
cmake ..
make
```

若使用 VSCode 并安装了合适的插件，那么在使用快捷键 `Ctrl+S` 保存 `CMakeLists.txt` 时，会自动生成生成项目构建所需的中间文件。

配置和生成过程如下图所示。

![](/assets/img/postsimg/20230515/cmake-configure.jpg)

生成完毕得到的中间文件如下图所示。

![](/assets/img/postsimg/20230515/cmake-configure-result.jpg)

### 构建（Build）

采用 CMake 构建项目有三种方式：
- 方式1：打开命令板（`Ctrl+Shift+P`）并运行 `CMake：Build`；
- 方式2：或从底部状态栏中点击"build"按钮；
- 方式3：打开命令行窗口（快捷键 `Ctrl + ~` ）输入 `cmake --build build`；

下图是采用方式2进行构建的示意图。

![](/assets/img/postsimg/20230515/build.png)

### 运行和调试

运行和调试项目，打开某个源代码文件，并设置一个断点。然后打开命令板（`Ctrl+Shift+P`），并运行 `CMake： dbug`，然后按F5继续调试。

或者点击 VSCode 下方的 【虫子】 图标进行 DEBUG 调试。

## 基于 CMake 的打包

### CPack

CPack 是 CMake 2.4.2 之后的一个内置工具，用于创建软件的二进制包和源代码包。

CPack 在整个 CMake 工具链的位置如下图所示。

![](/assets/img/postsimg/20230515/cpack.png)

CPack 支持打包的包格式有以下种类：

- 7Z (7-Zip file format)
- DEB (Debian packages)
- External (CPack External packages)
- IFW (Qt Installer Framework)
- NSIS (Null Soft Installer)
- NSIS64 (Null Soft Installer (64-bit))
- NuGet (NuGet packages)
- RPM (RPM packages)
- STGZ (Self extracting Tar GZip compression
- TBZ2 (Tar GZip compression)
- TXZ (Tar XZ compression)
- TZ (Tar Compress compression)
- ZIP (ZIP file format)

**为什么要用打包工具**：软件程序想要在生产环境快速被使用，就需要一个一键安装的安装包，这样生产环境就可以很方便的部署和使用。生成一键安装的安装包的工具就是打包工具。其中 NSIS 是 Windows 环境下使用的打包工具。

**选择 CPack 的原因**：C++ 工程大部分都是用 CMake 配置编译， 而 CPack 是 CMake 内置的工具，支持打包成多种格式的安装包。因为是 CMake 的内置工具，所以使用的方式也是通过在 CMakeLists.txt 配置参数，就能达到我们的需求。使用起来很方便，容易上手。

**如何安装 CPack**：安装 CMake 的时候会把 CPack 一起安装了。

### NSIS

官网下载最新版本并安装：https://nsis.sourceforge.io/Download。

NSIS是开发人员创建 Windows 下安装程序的工具。它可以创建能够安装、卸载、设置系统设置、提取文件等的安装程序。

NSIS允许您创建从只复制文件的基本安装程序到处理许多高级任务（如编写注册表项、设置环境变量、从internet下载最新文件、自定义配置文件等）的非常复杂的安装程序的所有内容。

NSIS基于脚本文件，支持变量、函数和字符串操作，就像一种普通的编程语言一样，但它是为创建安装程序而设计的。在默认选项下，它的开销只有34kb。同时由于其强大的脚本语言和对外部插件的支持，仍然提供了许多选项。

安装完成后，NSIS 具备一个 GUI，但是我一般不用，而是直接通过 CMakeLists.txt 文件调用 NSIS 进行打包。详见 [配置](#配置configure) 中的 `#打包`。

如果需要使用 GUI 来辅助生成打包脚本，参考 [此处](https://www.cnblogs.com/modou/p/3573772.html)。

### 基于 CPack 和 NSIS 的打包

完成项目构建后, 你会发现 `build` 目录下面多了两个文件 `CPackConfig.cmake` 和 `CPackSourceConfig.cmake`。在终端执行以下命令完成打包，得到可执行安装程序。

```
cpack.exe .\CPackConfig.cmake
```

![](/assets/img/postsimg/20230515/pack.jpg)

将安装程序分发到其它 Windows 平台即可完成安装。

## 参考文献

[1] maskerII. [【简书】CMakeLists 入门](https://www.jianshu.com/p/2bdcd7d7b164)

[2] TomKing-tm. [vsCode+CMake开发环境搭建](https://blog.csdn.net/weixin_43470971/article/details/119621643)

[3] 魔豆的BLOG. [使用NSIS制作安装包](https://www.cnblogs.com/modou/p/3573772.html)
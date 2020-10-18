---
title: 天文学基础（JPL星历）
date: 2020-10-18 21:49:19 +0800
categories: [Knowledge, Astronomy]
tags: [academic]
math: true
---

本文介绍了 JPL （美国喷气实验室）的星历（DE405）的基本概念，数据组织方式，以及具体的行星数据查询方法。

<!--more-->

---

- [1. JPL星历](#1-jpl星历)
  - [1.1. 概念](#11-概念)
  - [1.2. 版本](#12-版本)
  - [1.3. 算法](#13-算法)
- [行星位置计算](#行星位置计算)
  - [基本概念](#基本概念)
  - [DE405的结构](#de405的结构)
- [4. 参考文献](#4-参考文献)

# 1. JPL星历

## 1.1. 概念

JPL planetary ephemerides are generally created to support spacecraft missions to the planets. Selected ephemerides are recommended for more general use.

JPL星历表给出了太阳、月球和九大行星过去和将来的位置信息，并且是开放可使用的。JPL星历表在20世纪60年代由喷气推进实验室建立，最初用作行星探测导航的目的，随着观测技术的不断提高，新的观测数据不断获得，JPL星历表仍在不断修正和完善。
JPL星历表是对描述太阳系动力学系统的微分方程组进行数值积分的结果，因此它的建立基于两个假设：1.微分方程组精确代表了已知的动力学定律，至少在目前观测精度下是如此；2.数值积分程序精度足够高。在这两个假设的前提下，可以构建一个动力学系统，它与太阳系动力学系统是否一致还需要看初始条件和参数是否一致，初始条件和参数通过观测数据来进行最小二乘拟和。观测数据包括：行星探测飞船的测距数据、雷达行星测距数据、月球激光测距、光学观测的测角数据和一些最新的测量手段。
JPL星历表为了精确的表示长时间范围内的天体位置，把长时间范围（数百年）分成短的时间区间（数天），对于每个短的时间区间，它提供一组切比雪夫插值系数，要计算某一时刻的天体位置，首先找到这个短的时间区间，得到切比雪夫插值系数，然后根据切比雪夫插值公式计算天体位置。

## 1.2. 版本

The latest JPL ephemeris with fully consistent treatment of planetary and lunar laser ranging data is DE430 (Folkner et al 2014). The dynamical model for DE430 includes a frictional damping between the fluid core and the elastic mantle. This damping term is not suitable for extrapolation more than several centuries into the past. In order to cover a longer time span, the ephmeris DE431 was integrated without the lunar core/mantle damping term. The positions of the planets for DE431 agree with the positions on DE430 to within one meter over the time covered by DE430. For the Moon DE431 differs from DE430 mainly in the estimated tidal damping term causing a difference in along-track position of the Moon of 20 meters 100 years from the present and growing quadratically for times more thna 100 years from present.

## 1.3. 算法

The JPL planetary ephemerides are saved as files of Chebyshev polynomials fit to the Cartesian positions and velocities of the planets, Sun, and Moon, typically in 32-day intervals. The positions are integrated in astronomical units (au), but with polynomials stored in units of kilometers. The integration time units are days of barycentric dynamical time (TDB). Prior to DE430, the value of the astronomical unit was estimated from measurements of planetary orbits using the Gaussian gravitational constant k. Starting with DE430, the astronomical units has been fixed to the value 149597870.700 km as adopted by the International Astronomical Union in 2012.

《高健. 日月、行星位置计算》：JPL-DE405给出的是瞬时日月和行星在ICRS下的位置和速度

http://en.wikipedia.org/wiki/Jet_Propulsion_Laboratory_Development_Ephemeris

> DE405 was released in 1998. It added several years' extra data from telescopic, radar, spacecraft, and VLBI observations (of the Galileo spacecraft at Jupiter, in particular). The method of modeling the asteroids' perturbations was improved, although the same number of asteroids were modeled. The ephemeris was more accurately oriented onto the ICRF. DE405 covered 1600 to 2200 to full precision.

# 行星位置计算

## 基本概念

已知某一瞬时时刻（格里历日期+UTC时刻）
转换成儒略日（儒略纪元）
输入JPL-DE405，得到ICRS下的太阳、月球、各大行星的位置和速度
然后转至ITRS或J2000下。

两个关键点：

- JPL查表计算
- ICRS到ITRS或J2000的坐标变换。

## DE405的结构

根据创建时间不同JPL星历表有多个版本，这里采用DE405，它是1997年创建的，包括从1599年到2201年太阳系九大行星和月球的位置。DE405的文件包括头文件header.405和系数文件ascp****.405，****代表系数文件的起始时间，每个系数文件包含20年天体位置切比雪夫插值系数，例如从2000年到2020年的系数包含在文件ascp2000.405里。

DE405的头文件header.405包含了DE405的数据信息、天文常数和数据索引。数据索引是一个3行13列的表，每列数据代表一个天体的位置数据在数据块内的位置，依次为水星、金星、地月系统、火星、木星、土星、天王星、海王星、冥王星、月球、太阳，第12列数据代表章动角（nutations），包含两个角度：黄经章动 $\Psi$ 和交角章动 $\epsilon$ ，第13列数据代表岁差参数，包含三个欧拉角：$\zeta, z, \theta$ 。每列的第一行指示该天体数据在数据块的起始位置，第二行表示切比雪夫多项式的阶数，第三行表示该天体的数据被划分成几个子区间。


# 4. 参考文献

<span id="ref1">[1]</span> 高健. 《日月、行星位置计算》.

<span id="ref2">[2]</span> Wikipedia. [JPL Ephemeris](http://en.wikipedia.org/wiki/Jet_Propulsion_Laboratory_Development_Ephemeris).
---
title: CDDIS网站下GNSS相关数据解析（卫星星历部分）
date: 2021-11-18 17:05:49 +0800
categories: [Academic, Knowledge]
tags: [astronomy]
math: true
---

本文介绍了CDDIS网站下 GNSS 相关的数据产品下载、命名方式解读、文件格式说明和文件下载地址。

<!--more-->

 ---

- [1. 数据（data目录）](#1-数据data目录)
- [2. 广播星历（Broadcast ephemeris data）](#2-广播星历broadcast-ephemeris-data)
  - [2.1. Daily GPS Broadcast Ephemeris Files](#21-daily-gps-broadcast-ephemeris-files)
  - [2.2. Hourly GPS Broadcast Ephemeris Files](#22-hourly-gps-broadcast-ephemeris-files)
  - [2.3. Daily GLONASS Broadcast Ephemeris Files](#23-daily-glonass-broadcast-ephemeris-files)
- [3. 产品（product目录）](#3-产品product目录)
- [4. 精密星历](#4-精密星历)
  - [4.1. GPS 周](#41-gps-周)
- [5. 精密星历文件](#5-精密星历文件)
  

核心参考文献：[GNSS](https://blog.csdn.net/gou_hailong/category_10131532.html). [CDDIS网站下 GNSS 相关的数据产品下载+命名方式解读+文件格式说明文件下载地址](https://blog.csdn.net/Gou_Hailong/article/details/109191352)

## 1. 数据（data目录）

CDDIS存档包含来自永久GNSS接收器全球网络的GNSS数据，这些网络支持以30秒的采样率运行的IGS，并包含24小时的数据（UTC时间00：00-23：59）。 IGS分析中心每天都会检索这些数据以生产IGS产品。 这些产品，例如每日和每周的卫星星历，站的位置和速度，卫星和站的时钟信息以及地球自转参数，都将提交给CDDIS。

数据目录一览：

```
 **********************************************************************
                   Welcome to the CDDIS GPS Archive

The main directories are:
archive/gnss/data/
 /campaign                   GPS data from selected campaigns

 /daily/YYYY/DDD/YYd         Observation files for year YYYY and day DDD
                             (Hatanaka format)
                 /YYm        Met files for year YYYY and day DDD
                 /YYn        Navigation files for year YYYY and day DDD
                 /YYo        Observation files for year YYYY and day DDD
                 /YYs        Summary files for year YYYY and day DDD
                             (teqc output)
 /hourly/YYYY/DDD/HH         Hourly RINEX GPS files for year YYYY, day DDD,
                             and hour HH; observation (Hatanaka format),
                             navigation and met data

 /high-rate/YYYY/DDD/YYd/HH  Observation files for day YYDDD and hour HH
                             (Hatanaka format)
                    /YYm/HH  Met files for day YYDDD and hour HH
                    /YYn/HH  Navigation files for day YYDDD and hour HH

 /satellite/SATNAME/YYYY/DDD GPS data from on-board GPS receivers for
                             satellite SATNAME for year YYYY and day DDD

  /gnss/products/ionex/YYYY/DDD    Daily IONEX products for year YYYY and
                                  day DDD
               /trop/WWWW         Weekly GPS troposphere products for GPS
                                  week WWWW
                    /YYYY         Yearly GPS troposphere products for year
                                  YYYY 
                    /nrt/WWWW     Near real-time GPS troposphere products
                                  for GPS week WWWW
               /trop_cmp/YYYY/DDD Troposphere comparison results for year
                                  YYYY and day DDD
               /trop_new/YYYY/DDD New IGS troposphere product for year
                                  YYYY and day DDD
               /WWWW              Weekly GPS precise orbits, etc. for GPS
                                  week WWWW 

 All files are archived in UNIX compressed format.

 Contact Carey Noll (Carey.E.Noll@nasa.gov) for further information.
 **********************************************************************

 The local date and time is: %T
```

数据分为两种格式：
- RINEX V2 format： `YYYY/DDD/YYt/mmmmDDD#.YYt.Z`
- RINEX V3 format： `YYYY/DDD/YYt/XXXXMRCCC_K_YYYYDDDHHMM_01D_30S_tt.FFF.g`

RINEX V2 格式的每日 GNSS 数据使用 `mmmmDDD＃.YYt.Z` 文件名约定，并且为UNIX压缩格式。 从2016年的数据开始，所有使用RINEX V3文件命名约定和gzip压缩格式的RINEX V3格式的每日GNSS数据均以RINEX V2格式的数据归档在`/gnss/data/daily` 区域的子目录中。

**RINEX V2 格式文件名中字母的含义说明：**

`YYYY/DDD/YYt/mmmmDDD#.YYt.Z`

|code	|meaning|
| --- | --- |
|YYYY	|4位，代表年|
|DDD	|3位，年积日|
|YYt	  |YY为年份的后两位数字，t表示不同数据类型，具体含义如下|
||b = 组合广播星历数据|
||n = GPS广播星历数据|
||f = 北斗广播星历数据|
||g = GLONASS广播星历数据|
||l = Galileo广播星历数据|
||i = IRNSS广播星历数据|
||h = SBAS广播星历数据|
||q = QZSS广播星历数据，同d|
||m = 气象数据|
||d = 高压缩的观测数据，后续用 `crx2rnx.exe` 转换为o文件|
||o = 观测数据，同d|
||s = 观测摘要文件（TEQC的输出）|
||x = 混合广播星历数据|
|mmmm| 4位，IGS测站的名字|
|#|	1位，当一天中有多个文件的情况下，表示一天当中的第几个文件。通常为0表示一天当中的所有数据（一个文件）|
|.Z	 |UNIX压缩文件|

**RINEX V3 格式文件名中字母的含义说明：**

`YYYY/DDD/YYt/XXXXMRCCC_K_YYYYDDDHHMM_01D_30S_tt.FFF.g`

|code|	meaning|
| --- | --- |
|YYYY|	4位，代表年|
|DDD|	3位，年积日|
|YYt|	YY为年份的后两位数字，t表示不同数据类型，具体含义如下|
||b = 组合广播星历数据|
||d = 高压缩的观测数据，后续用crx2rnx.exe转换为o文件|
||f = 北斗广播星历数据|
||g = GLONASS广播星历数据|
||h = SBAS广播星历数据|
||i = IRNSS广播星历数据|
||l = Galileo广播星历数据|
||m = 气象数据|
||n = GPS广播星历数据|
||o = 观测数据，同d|
||q = QZSS广播星历数据|
||s = 观测摘要文件（TEQC的输出）|
||x = 混合广播星历数据|
|XXXX|	4位，IGS测站的名字|
|M|	标记编号(0-9)|
|R|	接收机编号(0 - 9)|
|CCC|	ISO国家代码|
|K|	数据来源，其中：|
||R =从使用供应商或其他软件的接收数据|
||S =从数据流(RTCM或其他)|
||U =未知|
|HH|	2位，小时|
|MM|	2位，分钟|
|tt|	数据类型，其中：|
||GO = GPS观测数据|
||RO = GLONASS观测数据|
||EO = Galileo观测数据|
||JO = QZSS观测数据|
||CO = BDS观测数据|
||IO = IRNSS观测数据|
||SO = SBAS观测数据|
||MO = 混合观测数据|
||GN = GPS导航数据|
||RN = GLONASS导航数据|
||EN = Galileo导航数据|
||JN = QZSS导航数据|
||CN = BDS导航数据|
||IN = IRNSS导航数据|
||SN = SBAS导航数据|
||MN = 导航数据(所有GNSS星座)|
||MM = 气象观测数据|
|FFF|	rnx = RINEX|
||crx = 高压缩的 RINEX|
|01D_30S|	一般来说，这个字段有三类：|
||01D_30S Daily 全天 采样30s|
||01H_30S hourly 整个小时 采样30s|
||15M_01S minutely 15分 采样1s|

## 2. 广播星历（Broadcast ephemeris data）

广播星历是接收机直接从天线接收到的卫星所发射的信号中分离出来的。

除了观测数据外，很大一部分的GNSS站点还提供广播导航数据。CDDIS从这些站点发送的这些特定于站点的文件中创建每日广播星历文件；这些文件（一个用于GPS，另一个用于GLONASS）包含每天的唯一GPS或GLONASS卫星星历消息。在UTC一天开始时会创建一个类似的文件，并从每小时广播的导航文件中每小时更新一次。因此，用户可以每天或每小时下载一个文件，其中包含后处理所需的所有广播星历消息。


### 2.1. Daily GPS Broadcast Ephemeris Files

每日GPS广播星历文件是将单个站点的导航文件合并成一个可供用户使用的非冗余文件，而不是多个单独的导航文件。每天在BKG创建的文件包含来自欧洲站点的独特导航消息。日常文件的起始目录为：

```
https://cddis.nasa.gov/archive/gnss/data/daily/
将以下目录和文件名附加到起始目录:
YYYY/DDD/YYn/brdcDDD0.YYn.Z (合并GPS广播星历文件)
OR
YYYY/brdc/brdcDDD0.YYn.Z (合并GPS广播星历文件)
YYYY/DDD/YYn/ifagDDD0.YYn.Z (以往在BKG创建的每日文件)
```

### 2.2. Hourly GPS Broadcast Ephemeris Files

组合的广播星历文件是每小时从CDDIS上存档的所有每小时导航文件中生成的。 每小时导航文件包含具有当天TOE的所有广播消息，该消息在小时的顶部创建时可用。 每小时使用新的导航消息更新文件。

在UTC一天结束时，当生成文件的最终版本时，该文件将复制到每日目录中，并成为“每日”广播星历表文件。

每小时文件的起始目录是

https://cddis.nasa.gov/archive/gnss/data/hourly/
将以下目录和文件名附加到起始目录:YYYY/DDD/hourDDDm.YYn.Z

### 2.3. Daily GLONASS Broadcast Ephemeris Files

类似地，可以在GLONASS导航文件子目录中找到每日仅使用GLONASS的广播星历文件。每日仅使用glonass文件的起始目录为

https://cddis.nasa.gov/archive/gnss/data/daily/
将以下目录和文件名附加到起始目录:
YYYY/DDD/YYg/brdcDDD0.YYg.ZOR
YYYY/brdc/brdcDDD0.YYg.Z

## 3. 产品（product目录）

产品目录一览：

https://cddis.nasa.gov/archive/gnss/products/

文件为：`WWWW/AAAWWWWD_TYP.YYt.Z`，其中：

|code|meaning|
| --- | --- |
| AAA | International GNSS Service(IGS) 分析中心名称 |
| WWWW | GPS 周 |
| D | 星期（0-6，7表示每周） |
| TYP | 解的类型，具体如下： |
|  | eph Satellite orbit solution 卫星轨道解 |
|  | erp Earth orientation parameter solution 地球定向参数解 |
|  | sp3 Satellite orbit solution 卫星轨道解 |
|  | sum Orbit solution analysis summary 轨道解分析总结 |
|.Z|	UNIX 压缩文件|

分析中心名称缩写对应如下：

|code|meaning|
| --- | --- |
| cod | CODE（3 day long arc solution） |
| cof | CODE（1 day solution） |
| cox | CODE GLONASS only |
| emr | NRCan |
| emx | NRCan GLONASS only |
| jpl | JPL |
| mit | MIT |
| ncl | University of Newcastle |
| sio | SIO |
等等不再详述。


## 4. 精密星历

精密星历由自建跟踪站提供（一般为当地、符合气象、大气层的实际）；可以较准确地提供轨道信息；事后计算；有偿服务；地面通讯获取。精密星历的获取需要根据GPS周来确定。

### 4.1. GPS 周

GPS周的计算方法参考：[流浪猪头拯救地球](https://blog.csdn.net/Gou_Hailong). [GPS周计算](https://blog.csdn.net/Gou_Hailong/article/details/100805581)。

GPS周（GPS Week）是GPS系统内部所采用的时间系统。 时间零点定义的为：1980年1月5日夜晚与1980年1月6日凌晨之间0点。最大时间单位是周（一周：604800秒）。由于在储存周数的时候，用了10个比特，2的10次方是1024，所以每1024周（即7168天）为一循环周期。

我们国家的北斗考虑到每1024周翻转一次过于频繁，所以就开了13个比特来存放周数，2的13次方为8192，大概是150多年翻转一次，我们这辈子恐怕没机会见到了，哈哈哈。

第一个GPS周循环点为1999年8月22日0时0分0秒。即从这一刻起，周数重新从0开始算起。星期记数规则是：Sunday为0，Monday为1，以此类推，依次记作0~6，GPS周记数为“GPS周 星期记数”。

GPS周与儒略日转换代码（https://github.com/fernandoferreiratbe/gpsweek-converter）：

```python
# _*_ encoding: utf-8 _*_

class Converter:

    # noinspection PyMethodMayBeStatic
    def convert_julian_date_to_gps_week(self, julian_date: float):
        gps_week = ((julian_date - 2444245) // 7)

        return int(gps_week)

    # noinspection PyMethodMayBeStatic
    def convert_gps_week_to_julian_date(self, gps_week: int, day_of_week: int = 0):
        if gps_week is None:
            raise ValueError('GPS Week can not be None.')

        if day_of_week not in range(0, 7):
            raise ValueError('Day of week out of range 0-6.')

        julian_day = (7.0 * gps_week + 2444245.0 + day_of_week)

        return julian_day
```

或者采用在线计算的方式（https://www.labsat.co.uk/index.php/en/gps-time-calculator）。

比如想找2020年元旦的精密星历数据，经过计算知道那一天是GPS周第2086周，所以进入2086目录下去下载相应数据。

## 5. 精密星历文件

目前IGS精密星历主要分为三种：最终精密星历（IGS Final，标识为 IGS）、快速精密星历（IGS Rapid，标识为 IGR）、以及超快速精密星历（IGS Ultra-Rapid，标识为 IGU）。对应的精密钟差也有这三种。其中超快速精密星历又分为观测的部分和预测的部分。

IGS 会综合所有分析中心的产品（比如SIO，MIT等）加权平均得到最终的产品（标识为IGS、IGR、IGU）。他们的延时、精度等指标如下表所示。在实际工作中，我们可以根据项目对时间及精度的要求，选取不同类型的文件来使用。

|名称|延时|更新率|更新时间|采样率|精度|
|---|---|---|---|---|---|
|最终精密星历 IGS|12-18天|每周|每周四|15min|~2.5cm|
|快速精密星历 IGR|17-41小时|每天|17:00 UTC|15min|~2.5cm|
|超快速精密星历（观测） IGU|3-9小时|6小时|03, 09, 15, 21 UTC|15min|~3cm|
|超快速精密星历（预测） IGU|实时|6小时|03, 09, 15, 21 UTC|15min|~5cm|


IGS 精密星历采用sp3格式，sp3文件的存储方式为 ASCII 文本文件，内容包括表头信息以及文件体，文件体中每隔15分钟给出卫星的位置，有时还给出卫星的速度。如果需要其他时刻的卫星位置，可以由给出的卫星位置进行插值得到。关于sp3文件的详细说明可以参考官方文档。

与广播星历不一样，精密星历并不是用开普勒参数给出的，而是直接给出卫星在 ITRF框架中的坐标值 ITRF（International Terrestrial Reference Frame 国际地球参考框架）框架是由 IERS（国际地球自转服务）发布的，ITRF 的构成是基于VLBI、LLR、SLR、GPS 和 DORIS 等空间大地测量技术的观测数据，由 IERS中心局分析得到的全球站坐标和速度场。自1988 年起，IERS 已经发布 ITRF88、ITRF89、ITRF90、ITRF9 1、ITRF92、ITRF93、ITRF94、ITRF96、ITRF2000、ITRF2005、ITRF2008、ITRF2014 等全球参考框架。

下面是从igs20863.sp3文件中截取的开头部分：

```
#cP2020  1  1  0  0  0.00000000      96 ORBIT IGS14 HLM  IGS
## 2086 259200.00000000   900.00000000 58849 0.0000000000000
+   32   G01G02G03G04G05G06G07G08G09G10G11G12G13G14G15G16G17
+        G18G19G20G21G22G23G24G25G26G27G28G29G30G31G32  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
++         2  2  2  2  2  2  2  2  3  2  2  2  2  2  2  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* FINAL ORBIT COMBINATION FROM WEIGHTED AVERAGE OF:        
/* cod emr esa gfz grg jpl mit ngs sio                      
/* REFERENCED TO IGS TIME (IGST) AND TO WEIGHTED MEAN POLE:
/* PCV:IGS14_2086 OL/AL:FES2004  NONE     Y  ORB:CMB CLK:CMB
*  2020  1  1  0  0  0.00000000
PG01   9888.347661 -19617.637150 -14892.695413   -247.874423 12 10  5  71       
PG02 -21823.783543  13971.156772  -4920.780210   -377.229970 10  9  7  81       
PG03  -1601.529966 -16464.253025 -20857.441401    -63.370799  8  7 11 105       
PG04   1221.994671 -26086.976634  -4781.101677    -27.923755 12  4  5 109       
PG05 -19154.192243   5721.537965  17444.937173     -5.436142  7  8  6  88       
PG06 -21150.813850   2402.751626 -15853.420427   -169.653500  9  7  8 106       
PG07  -7307.639840 -13757.950583  21767.441209   -182.185020  6  6  5  99       
PG08   7039.047666 -21245.446842  14152.046835    -18.646350  8  5  7  94       
PG09  -6875.945629 -25134.252934   5050.629544   -121.818618  9  8  4 109       
PG10  24015.414289  11544.134960   1555.732288   -193.008701  8  8  8 103       
PG11   7282.297820 -25005.780611  -6416.591136   -403.117157  9  8  7  95       
PG12  -3612.078652  14754.643360 -22024.819114    167.238453  6  6  5 121       
PG13 -12822.251931  14077.120706  18359.989020    -22.958664  7  7  7  85       
PG14  15097.782136    719.539797 -21518.284406    -41.480187  8  9  3  98       
PG15  -5026.428258  22304.034935  12884.805910   -261.734981  8  6  5  83       
PG16  20065.806133  -2268.606859  17288.279253   -108.806027  7  8  6  98       
PG17 -13302.574437 -12501.633252 -18888.874080    190.733864  6  5  7  99       
PG18  15683.730129 -20086.062189  -7815.214505 999999.999999
PG19 -14744.295897  -4057.283573 -22007.637771   -218.643006  9  7  9 107       
PG20  18932.638979  14547.678555  11749.766535    527.763846 11 10 11  93       
PG21  10191.957706  12064.511376  22150.584734    -62.423168 10  8 11  69       
PG22   7023.579706 -14225.418680 -21077.053685   -780.142685  9  7  9  70       
PG23  -1206.975660 -26081.394867  -3218.011480   -145.988210 10  4  6 100       
PG24 -10589.076575  20789.036009 -12534.802874     -6.933769  9  7  4 109       
PG25  11115.892105  17563.783060 -16907.847503    -16.767205  8  8  6 106       
PG26  24809.687134   2652.030547   9387.846884    117.742691  8 10  8 116       
PG27  12654.914692 -10785.577586  20515.681941   -167.508526  8  5  8  92       
PG28 -22961.626731 -13197.537237    783.057544    747.765166  6  8  6  95       
PG29   4498.761624  25678.893062   4992.306129      4.083857 10 10  6  67       
PG30 -17217.300944  -5541.835900  19540.416589   -118.767102  6  7  7  87       
PG31  22625.699147  -4025.359523 -13738.508642    -13.516273  7  9  7  90       
PG32  14835.028318  11206.018688 -18957.170663    172.050190  9 10  5 104       
*  2020  1  1  0 15  0.00000000
PG01  11218.886280 -20435.428221 -12620.054414   -247.885427 11 10  5  55       
PG02 -21416.153632  13234.932709  -7652.201926   -377.236434  9  9  6  89       
PG03    658.942604 -15624.793321 -21537.118493    -63.377346  7  7 10  93       
PG04   1551.226922 -25415.284695  -7518.401629    -27.928395 11  5  6  93       
PG05 -20891.993883   4906.521003  15635.291423     -5.436689  7  8  5  80       
PG06 -19726.065274   1101.751537 -17715.564454   -169.663056  9  6  7 113
... ... ...
```

简单介绍一下
- 第一行，时间，2020年1月1日，IGS14 表示座标系   采用ITRF2014框架。
- 第二行，2086表示GPS周，259200表示累计秒数（3天*86400）
- 第三/四行，包含32颗GPS卫星的轨道位置
- 第23行表示从第24行开始的这一组卫星轨道位置对应的时间
- 第56行表示从第57行开始的这一组卫星轨道位置对应的时间，与前一数据块相差15分钟
- 第24行为第一个卫星的位置和钟差信息，位置单位为km，精度到mm，包含xyz三个方向。时钟单位为ms，精度为ps。最后四个数为xyz位置和钟差的标准差。
- 标准差的值要结合第15行的浮点基准值来计算，比如第一颗卫星x方向标准差为12，结合第15行第一个浮点基准值1.25，标准差计算为 $1.25^{12} = 14.5519 mm$。钟差计算方法类似，不过用的第二个基准值1.025。

[官网](https://cddis.nasa.gov/Data_and_Derived_Products/CDDIS_Archive_Access.html)有抓取爬虫的代码和教程。示例代码如下：

```python
import requests

url='https://cddis.nasa.gov/archive/gnss/products/'
dest = 'C:/GPS/'

### determine GPS week range
GPSweek_start = 2086
GPSweek_end = 2087

for i in range(GPSweek_end-GPSweek_start):
    gpsw = str(GPSweek_start+i)
    for d in range(7):
        filename = '/igs'+gpsw+str(d)+'.sp3.Z' # e.g., igs20860.sp3.Z
        path = url + gpsw + filename
        r=requests.get(path)
        destf = dest + filename
        with open(destf, "wb") as f:
            for chunk in r.iter_content(chunk_size=1000):
                f.write(chunk)
        f.close()
```

注意需要将 `.netrc` 文件放在指定位置后，将该位置加入到系统环境变量中，变量名为 `HOME`。然后采用 python 即可抓取数据。同样也可以采用cURL 工具，请参考[这里](https://navrs.wh.sdu.edu.cn/info/1621/1487.htm)。
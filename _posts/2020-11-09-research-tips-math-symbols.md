---
title: 科研Tips（数学符号）
date: 2020-11-09 10:01:19 +0800
categories: [Tutorial, Writing]
tags: [academic]
math: true
---

本文列举了常用数学符号以供平时查询，包括希腊字母、二元关系符、二元运算符。

<!--more-->
- [1. 希腊字母](#1-希腊字母)
- [2. 二元关系符](#2-二元关系符)
- [3. 二元运算符](#3-二元运算符)
- [4. 大尺寸运算符](#4-大尺寸运算符)
- [5. 箭头](#5-箭头)
- [6. 其它符号](#6-其它符号)
---

# 1. 希腊字母


大写命令 | 大写 | 小写命令 | 小写
-|-|-|-
`\Alpha` | $\Alpha$ | `\alpha` | $\alpha$
`\Beta` | $\Beta$ | `\beta` | $\beta$
`\Gamma` | $\Gamma$ | `\gamma` | $\gamma$
`\Delta` | $\Delta$ | `\delta` | $\delta$
`\Epsilon` | $\Epsilon$ | `\epsilon,\varepsilon` | $\epsilon,\varepsilon$
`\Zeta` | $\Zeta$ | `\zeta` | $\zeta$
`\Eta` | $\Eta$ | `\eta` | $\eta$
`\Theta` | $\Theta$ | `\theta` | $\theta$
`\Rho` | $\Rho$ | `\rho,\varrho` | $\rho,\varrho$
`\Sigma` | $\Sigma$ | `\sigma` | $\sigma$
`\Tau` | $\Tau$ | `\tau` | $\tau$
`\Upsilon` | $\Upsilon$ | `\upsilon` | $\upsilon$
`\Phi` | $\Phi$ | `\phi,\varphi` | $\phi,\varphi$
`\Chi` | $\Chi$ | `\chi` | $\chi$
`\Psi` | $\Psi$ | `\psi` | $\psi$
`\Omega` | $\Omega$ | `\omega` | $\omega$

# 2. 二元关系符

命令 | 符号 ||命令| 符号||命令| 符号
-|-|-|-|-|-|-|-
`\leq, \le` | $\leq$ ||`\geq, \ge`| $\geq$||`\equiv`| $\equiv$
`\ll`| $\ll$ ||`\gg`| $\gg$||`\doteq`| $\doteq$
`\prec`| $\prec$ ||`\succ`| $\succ$||`\sim`| $\sim$
`\preceq`| $\preceq$ ||`\succeq`| $\succeq$||`\simeq`| $\simeq$
`\subset`| $\subset$ ||`\supset`| $\supset$||`\approx`| $\approx$
`\subseteq`| $\subseteq$ ||`\supseteq`| $\supseteq$||`\cong`| $\cong$
`\in`| $\in$ ||`\ni,\owns`| $\ni$||`\propto`| $\propto$
`\vdash`| $\vdash$ ||`\dashv`| $\dashv$||`\models`| $\models$
`\mid`| $\mid$ ||`\parallel`| $\parallel$||`\perp`| $\perp$
`\smile`| $\smile$ ||`\frown`| $\frown$||`\asymp`| $\asymp$
`:`| $:$ ||`\notin`| $\notin$||`\neq,\ne`| $\neq$

注意，有 3 个比较特殊的关系符，在使用时需要添加 `latexsym` 宏包：

命令 | 符号 ||命令| 符号||命令| 符号
-|- |-|-|-|-|-|-
`\sqsubseteq`| $\sqsubseteq$ ||`\sqsupseteq`| $\sqsupseteq$||`\bowtie`| $\bowtie$

# 3. 二元运算符

命令 | 符号 ||命令| 符号||命令| 符号
-|-|-|-|-|-|-|-
`\pm` | $\pm$ ||`\mp`| $\mp$||`\triangleleft`| $\triangleleft$
`\cdot`| $\cdot$ ||`\div`| $\div$||`\triangleright`| $\triangleright$
`\times`| $\times$ ||`\setminus`| $\setminus$||`\star`| $\star$
`\cup`| $\cup$ ||`\cap`| $\cap$||`\ast`| $\ast$
`\sqcup`| $\sqcup$ ||`\sqcap`| $\sqcap$||`\circ`| $\circ$ (上标可作度$^{\circ}$)
`\vee, \lor`| $\vee$ ||`\wedge, \land`| $\wedge$||`\bullet`| $\bullet$
`\oplus`| $\oplus$ ||`\ominus`| $\ominus$||`\diamond`| $\diamond$
`\odot`| $\odot$ ||`\oslash`| $\oslash$||`\uplus`| $\uplus$
`\otimes`| $\otimes$ ||`\bigcirc`| $\bigcirc$||`\amalg`| $\amalg$
`\bigtriangleup`| $\bigtriangleup$ ||`\bigtriangledown`| $\bigtriangledown$||`\dagger`| $\dagger$
`\ddagger`| $\ddagger$ ||`\wr`| $\wr$||| 

其中，有 4 个特殊符号需要添加 `latexsym` 宏包：

命令 | 符号 ||命令| 符号
-|-|-|-|-
`\lhd` | $\lhd$ ||`\rhd`| $\rhd$
`\unlhd`| $\unlhd$ ||`\unrhd`| $\unrhd$

# 4. 大尺寸运算符

命令 | 符号 ||命令| 符号||命令| 符号
-|-|-|-|-|-|-|-
`\sum` | $\sum$ ||`\bigcup`| $\bigcup$||`\bigvee`| $\bigvee$
`\prod`| $\prod$ ||`\bigcap`| $\bigcap$||`\bigwedge`| $\bigwedge$
`\coprod`| $\coprod$ ||`\bigsqcup`| $\bigsqcup$||`\biguplus`| $\biguplus$
`\bigoplus`| $\bigoplus$ ||`\bigotimes`| $\bigotimes$||`\bigodot`| $\bigodot$
`\int`| $\int$ ||`\oint`| $\oint$|||

# 5. 箭头

命令 | 符号 ||命令| 符号||命令| 符号
-|-|-|-|-|-|-|-
`\leftarrow, \gets` | $\leftarrow$ ||`\longleftarrow`| $\longleftarrow$||`\uparrow`| $\uparrow$
`\rightarrow, \to`| $\rightarrow$ ||`\longrightarrow`| $\longrightarrow$||`\downarrow`| $\downarrow$
`\leftrightarrow`| $\leftrightarrow$ ||`\longleftrightarrow`| $\longleftrightarrow$||`\updownarrow`| $\updownarrow$
`\Leftarrow`| $\Leftarrow$ ||`\Longleftarrow`| $\Longleftarrow$||`\Uparrow`| $\Uparrow$
`\Rightarrow`| $\Rightarrow$ ||`\Longrightarrow`| $\Longrightarrow$||`\Downarrow`| $\Downarrow$
`\Leftrightarrow`| $\Leftrightarrow$ ||`\Longleftrightarrow`| $\Longleftrightarrow$||`\Updownarrow`| $\Updownarrow$
`\mapsto`| $\mapsto$ ||`\longmapsto`| $\longmapsto$||`\nearrow`| $\nearrow$
`\hookleftarrow`| $\hookleftarrow$ ||`\hookrightarrow`| $\hookrightarrow$||`\searrow`| $\searrow$
`\leftharpoonup`| $\leftharpoonup$ ||`\rightharpoonup`| $\rightharpoonup$||`\swarrow`| $\swarrow$
`\leftharpoondown`| $\leftharpoondown$ ||`\rightharpoondown`| $\rightharpoondown$||`\nwarrow`| $\nwarrow$
`\rightleftharpoons`| $\rightleftharpoons$ ||`\iff`| $\iff$|| `\`|$\backslash$

其中，有 1 个特殊符号需要添加 `latexsym` 宏包：

命令 | 符号
-|-
`\leadsto` | $\leadsto$

# 6. 其它符号

命令 | 符号 ||命令| 符号||命令| 符号
-|-|-|-|-|-|-|-
`\cdots` | $\cdots$ ||`\vdots`| $\vdots$||`\ddtos`| $\ddots$
`\hbar`| $\hbar$ ||`\ell`| $\ell$||`\Re`| $\Re$
`\aleph`| $\aleph$ ||`\forall`| $\forall$||`\partial`| $\partial$
`\nabla`| $\nabla$ ||`\infty`| $\infty$||`\empty`| $\empty$
`\bot`| $\bot$ ||`\top`| $\top$||`\varnothing`| $\varnothing$
`\flat`| $\flat$ ||`\natural`| $\natural$||`\sharp`| $\sharp$
`\prime`| $\prime$ ||`\exists`| $\exists$||`\angle`| $\angle$

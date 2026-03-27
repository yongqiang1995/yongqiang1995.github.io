---
layout: post
title: "函数求导法则"
subtitle: "常见求导公式与复合函数求导"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
mathjax: true
tags:
  - 数学基础
  - 微积分
---

## 导数的定义

导数 (Derivative) 描述的是函数在某一点处的瞬时变化率. 设函数 $y = f(x)$ 在点 $x$ 的某个邻域内有定义, 当自变量 $x$ 在 $x$ 处取得增量 $h$ (即 $\Delta x = h$) 时, 如果极限

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

存在, 则称函数 $f(x)$ 在点 $x$ 处可导, 该极限值即为 $f(x)$ 在 $x$ 处的导数, 记作 $f'(x)$ 或 $\frac{df}{dx}$.

从几何意义上看, $f'(x_0)$ 表示曲线 $y = f(x)$ 在点 $(x_0, f(x_0))$ 处切线的斜率.

## 基本求导公式

以下是常用的基本初等函数求导公式:

| 函数 $f(x)$ | 导数 $f'(x)$ | 说明 |
|:---:|:---:|:---|
| $c$ (常数) | $0$ | 常数的导数为零 |
| $x^n$ | $n x^{n-1}$ | 幂函数求导 (Power Rule) |
| $e^x$ | $e^x$ | 自然指数函数 |
| $a^x$ | $a^x \ln a$ | 一般指数函数 ($a > 0, a \neq 1$) |
| $\ln x$ | $\frac{1}{x}$ | 自然对数函数 |
| $\log_a x$ | $\frac{1}{x \ln a}$ | 一般对数函数 |
| $\sin x$ | $\cos x$ | 正弦函数 |
| $\cos x$ | $-\sin x$ | 余弦函数 |
| $\tan x$ | $\sec^2 x$ | 正切函数 |
| $\frac{1}{x}$ | $-\frac{1}{x^2}$ | 倒数函数 |
| $\sqrt{x}$ | $\frac{1}{2\sqrt{x}}$ | 平方根函数 |

## 求导法则

### 和差法则 (Sum / Difference Rule)

$$
[f(x) \pm g(x)]' = f'(x) \pm g'(x)
$$

**示例**: 设 $y = x^3 + \sin x$, 则 $y' = 3x^2 + \cos x$.

### 常数倍法则 (Constant Multiple Rule)

$$
[c \cdot f(x)]' = c \cdot f'(x)
$$

### 乘法法则 (Product Rule)

$$
[f(x) \cdot g(x)]' = f'(x) \cdot g(x) + f(x) \cdot g'(x)
$$

**示例**: 设 $y = x^2 \cdot e^x$, 则:

$$
y' = 2x \cdot e^x + x^2 \cdot e^x = (2x + x^2) e^x
$$

### 除法法则 (Quotient Rule)

$$
\left[\frac{f(x)}{g(x)}\right]' = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}, \quad g(x) \neq 0
$$

**示例**: 设 $y = \frac{x}{\sin x}$, 则:

$$
y' = \frac{1 \cdot \sin x - x \cdot \cos x}{\sin^2 x} = \frac{\sin x - x \cos x}{\sin^2 x}
$$

### 链式法则 (Chain Rule)

若 $y = f(u)$, $u = g(x)$, 即 $y = f(g(x))$, 则:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(u) \cdot g'(x)
$$

链式法则是复合函数求导的核心, 也是深度学习反向传播的数学基础.

## 复合函数求导

复合函数的求导需要逐层应用链式法则, 将外层函数对内层函数的导数与内层函数对自变量的导数相乘.

### 示例 1

设 $y = e^{3x^2}$, 令 $u = 3x^2$, 则 $y = e^u$:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = e^u \cdot 6x = 6x \cdot e^{3x^2}
$$

### 示例 2

设 $y = \ln(\sin x)$, 令 $u = \sin x$, 则 $y = \ln u$:

$$
\frac{dy}{dx} = \frac{1}{u} \cdot \cos x = \frac{\cos x}{\sin x} = \cot x
$$

### 示例 3: 多层复合

设 $y = e^{\sin(x^2)}$, 令 $u = \sin(x^2)$, $v = x^2$:

$$
\frac{dy}{dx} = e^{\sin(x^2)} \cdot \cos(x^2) \cdot 2x
$$

## 在深度学习中的应用

在深度学习中, **反向传播算法 (Backpropagation)** 的核心就是链式法则. 神经网络由多层函数复合而成, 计算损失函数对每个参数的梯度时, 需要从输出层逐层向输入层应用链式法则.

以 Sigmoid 激活函数为例, 其导数可以用自身表示:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

这一性质使得梯度计算非常高效 —— 只需利用前向传播中已经计算好的 $\sigma(z)$ 值即可得到导数. 关于 Sigmoid 函数的详细介绍, 请参考激活函数之 Sigmoid 函数一文.

## 常用激活函数导数表

下表总结了深度学习中常见激活函数及其导数, 它们在反向传播时被频繁使用:

| 激活函数 | 表达式 $f(x)$ | 导数 $f'(x)$ |
|:---:|:---:|:---:|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ |
| ReLU | $\max(0, x)$ | $\begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}$ |
| Leaky ReLU | $\begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases}$ | $\begin{cases} 1, & x > 0 \\ \alpha, & x \leq 0 \end{cases}$ |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $S_i(\delta_{ij} - S_j)$ |

其中 Softmax 的导数涉及 Kronecker delta $\delta_{ij}$, 当 $i = j$ 时 $\delta_{ij} = 1$, 否则为 $0$.

## 参考

- 同济大学数学系, 《高等数学》第七版
- Goodfellow et al., *Deep Learning*, Chapter 6: Deep Feedforward Networks


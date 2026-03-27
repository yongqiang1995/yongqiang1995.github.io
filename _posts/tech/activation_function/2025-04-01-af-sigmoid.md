---
layout: post
title: "激活函数之 <code> Sigmoid </code> 函数"
subtitle: "十分钟深入理解 Sigmoid 函数的设计原理"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
mathjax: true
tags:
  - 激活函数
  - Sigmoid
---


# 和 Softmax 区别

![](/img/in-post/post-af-sigmoid/image.png)

### Sigmoid 函数详解

> 一种 `非线性` 激活函数, 常用于求解输出结果的概率分布.

#### 函数曲线

![](/img/in-post/Sigmoid_function.png)

从上图可以直观得到以下信息:

- `Sigmoid` 函数的值域范围在 `0~1` 之间, 当 `x=0` 时候, `y=0.5`
- 当自变量趋于正负无穷时, 函数的导数或者说梯度接近 `0`

#### 函数定义

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

也可以写成 $\sigma{z} = (1 + e^{-z})^{-1}$

其中:

- $\sigma$	=	Sigmoid

- $\vec{z}$	=	input vector

- $e^{z_{i}}$	=	standard exponential function for input vector

- $i$ = input vector 中的第 `i` 个元素

- $K$	=	number of classes in the multi-class classifier

- $e^{z_{j}}$	=	standard exponential function for output vector

- $j$ = 多分类任务中第 `j` 个分类

## 函数求导

$\sigma(z)$ 导数如下:

$$
\sigma(z)' = \sigma(z)(1 - \sigma(z)) \tag{1}
$$

根据复合函数求导法则, 求导过程如下:

$$
\begin{align}
\sigma(z)' =& -1 · (1 + e^{-z})^{-2} · (-1 · e^{-z}) \\
=& (1 + e^{-z})^{-2} · ((1 + e^{-z}) -1) \\
=&  (1 + e^{-z})^{-1} - (1 + e^{-z})^{-2} \\
=& (1 + e^{-z})^{-1}·(1 - (1 + e^{-z})^{-1}) \\
=& \sigma(z)·(1 - \sigma(z))
\end{align}
$$


#### 计算过程

假设输入的 `input_vector` (即上文公式中的 $\vec{z}$) 为 `[1, 2, 3, 4, 5]`, `shape` 为 `(5,)`. Sigmoid 函数对每个元素独立计算, 将其映射到 $(0, 1)$ 区间.

根据 Sigmoid 公式 $\sigma(z) = \frac{1}{1 + e^{-z}}$, 对每个元素分别计算:

- $\sigma(1) = \frac{1}{1 + e^{-1}} \approx 0.7311$
- $\sigma(2) = \frac{1}{1 + e^{-2}} \approx 0.8808$
- $\sigma(3) = \frac{1}{1 + e^{-3}} \approx 0.9526$
- $\sigma(4) = \frac{1}{1 + e^{-4}} \approx 0.9820$
- $\sigma(5) = \frac{1}{1 + e^{-5}} \approx 0.9933$

最终 `input_vector` 对应的 `Sigmoid` 结果为 `[0.7311, 0.8808, 0.9526, 0.9820, 0.9933]`.

注意: 与 Softmax 不同, Sigmoid 对每个元素独立计算, 输出值之间没有归一化关系, 各输出之和不一定为 1.

#### 输入示例

分别以 `numpy` 实现和 `torch` 实现进行比较, 可以发现二者的输出结果一致 (注意: `torch` 输出做了截断).

```py
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

vector_z = np.array([1, 2, 3, 4, 5])
print(f"sigmoid(vector_z)=", sigmoid(vector_z))

# 输出
# sigmoid(vector_z)= [0.73105858 0.88079708 0.95257413 0.98201379 0.99330715]

import torch

vector_z = torch.from_numpy(vector_z).to(dtype=torch.float32)
print(f"sigmoid(vector_z)=", torch.sigmoid(vector_z))

# 输出
# sigmoid(vector_z)= tensor([0.7311, 0.8808, 0.9526, 0.9820, 0.9933])
```

#### 设计原理

Sigmoid 函数的设计源自对数几率 (log-odds) 的数学关系. 如果某事件发生的概率为 $p$, 则其几率 (odds) 为 $\frac{p}{1-p}$, 对数几率为 $\ln\frac{p}{1-p}$. 令对数几率等于输入 $z$:

$$
\ln\frac{p}{1-p} = z \quad \Rightarrow \quad p = \frac{1}{1 + e^{-z}} = \sigma(z)
$$

因此 Sigmoid 本质上是**对数几率函数的逆函数**, 它将任意实数值 $z \in (-\infty, +\infty)$ 映射到概率区间 $(0, 1)$. 这一性质使其天然适合用于二分类问题中的概率输出.

选择指数函数 $e^{-z}$ 的原因在于其良好的数学性质: 处处可导、严格单调, 且导数可以用函数自身简洁表示 $\sigma'(z) = \sigma(z)(1-\sigma(z))$, 极大地简化了梯度计算.



---
layout: post
title: "激活函数之 <code> Softmax </code> 函数 NEW"
subtitle: "十分钟深入理解 softmax 函数的设计原理"
author: "YongQiang"
header-style: text
hidden: true
published: false
catalog: true
mathjax: true
tags:
  - 激活函数
  - SoftMax
---

### Softmax 函数详解

#### 函数简介

> 在数学(尤其是概率论及其相关学科)领域中, `Softmax` 函数, 或称 `归一化指数函数`, 是 [逻辑斯谛函数](https://zh.wikipedia.org/zh-cn/%E9%80%BB%E8%BE%91%E6%96%AF%E8%B0%9B%E5%87%BD%E6%95%B0) 的一种推广.

`Softmax` 函数是一种 `多元非线性` 激活函数, 常用于求解输出结果的概率分布. 与其它单变量非线性激活函数 (例如 `ReLU`, `Tanh`, `Sigmoid` 等) 不属于同一类别.

`多元` 也就意味着没有一个准确的曲线可以完整表达该函数.

`非线性` 意味着 (?)

#### 函数功能

`Softmax` 能将一个含任意实数的 `K维` 向量 ${\displaystyle \mathbf{z}}$ "压缩" 到另一个 `K维` 实向量 ${\displaystyle \sigma (\mathbf{z})}$ 中, 使得每一个元素的范围都在 ${\displaystyle (0, 1)}$ 之间, 并且所有元素的和为 `1`.

给定单个 `5维` 输入向量 $\mathbf{z}$, 经过 `softmax` 函数得到另一个 `5维` 向量, 示例如下:

$$
\mathbf{z} = [1, 2, 3, 4, 5] \rightarrow \sigma(\mathbf{z}) = [0.0117, 0.0317, 0.0861, 0.2341, 0.6364]
$$

向量 $\mathbf{z}$ 的 `shape` 为 `(5,)`. 在多分类任务中, `shape` 中的 `5` 可以表示五种分类, 经过 `Softmax` 之后, 其中的每一个元素表示输入数据属于该分类的概率大小.

#### 函数曲线 (2D 数据可视化)

绘制 `2D+` 的 `Softmax` 函数曲线可能并不容易, 但绘制 `2D-Softmax` 函数曲线用来说明问题却非常简单.

![](/img/in-post/softmax_function.png)

从上图可以直观得到以下信息:

- `softmax` 函数的值域范围在 `0~1` 之间
- 当自变量趋于正负无穷时, 函数的导数或者说梯度接近 `0`

#### 函数定义

$$
\sigma(\vec{z})_{i} = \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}
$$

其中:

- $\sigma$	=	softmax

- $\vec{z}$	=	input vector

- $e^{z_{i}}$	=	standard exponential function for input vector

- $i$ = input vector 中的第 `i` 个元素

- $K$	=	number of classes in the multi-class classifier

- $e^{z_{j}}$	=	standard exponential function for output vector

- $j$ = 多分类任务中第 `j` 个分类


#### 计算过程

以 `input_vector` (即上文公式中的 $\vec{z}$) `5D` 向量 `[1, 2, 3, 4, 5]` 作为输入讲解如何计算 `Softmax`.

首先对 `input_vector` 中每个元素计算 `e` 指数.

- 例如取第一个元素: $input\\_vector[0]$ 为 1
- $e^{z_i}$ 当 $i=0$ 时, $e^{input\\_vector[0]}$ 即 $e^1$ 的计算结果为 $e$

同理可以计算出其他元素对应的 `e` 指数结果:

- $i=1$ 时, 结果为 $e^2$
- $i=2$ 时, 结果为 $e^3$
- $i=3$ 时, 结果为 $e^4$
- $i=4$ 时, 结果为 $e^5$

计算 `input_vector` 中所有元素对应的 `e` 指数之和.

- $e\\_sum = e^1 + e^2 + e^3 + e^4 + e^5$

结果为 `233.2041839862982`.

计算 `softmax` 结果:

- $e^1 / e\\_sum$ 即 `0.011656230956039607`
- $e^2 / e\\_sum$ 即 `0.03168492079612427`
- $e^3 / e\\_sum$ 即 `0.0861285444362687`
- $e^4 / e\\_sum$ 即 `0.23412165725273662`
- $e^5 / e\\_sum$ 即 `0.6364086465588308`

最终 `input_vector` 对应的 `Softmax` 结果为 `[0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865]`.

#### 输入示例

分别以 `numpy` 实现和 `torch` 实现进行比较, 可以发现二者的输出结果一致 (注意: `torch` 输出做了截断).

```py
import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

vector_z = np.array([1, 2, 3, 4, 5])
print(f"softmax(vector_z)=", softmax(vector_z))

# 输出
# softmax(vector_z)= [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]

import torch
import torch.nn.functional as F

vector_z = torch.from_numpy(vector_z).to(dtype=torch.float32)
print(f"softmax(vector_z)=", F.softmax(vector_z, dim=0))

# 输出
# softmax(vector_z)= tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])
```

#### 设计原理


#### SoftMax 求导过程



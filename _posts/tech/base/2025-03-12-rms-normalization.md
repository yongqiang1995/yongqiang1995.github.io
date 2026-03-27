---
layout: post
title: "理解 <code>RMSNorm</code>"
subtitle: "Root Mean Square Layer Normalization 原理与实现"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
mathjax: true
tags:
  - Normalization
  - 深度学习基础
---

## RMSNorm 介绍

**RMSNorm**（Root Mean Square Layer Normalization）由 Zhang & Sennrich 在 2019 年提出，是 LayerNorm 的一种简化变体。

LayerNorm 包含两个操作：**re-centering**（减去均值）和 **re-scaling**（除以标准差）。RMSNorm 的核心发现是：re-centering 对模型性能的贡献并不显著，真正重要的是 re-scaling 操作。因此，RMSNorm 直接去掉了均值计算，只保留了基于均方根（RMS）的缩放。

这一简化使得 RMSNorm 在计算上更高效，同时不牺牲模型精度。目前，RMSNorm 已被广泛应用于主流大语言模型中，包括 **LLaMA**、**Gemma**、**Qwen**、**Mistral** 等。

## 数学公式

给定输入向量 $x = (x_1, x_2, \dots, x_n)$，RMSNorm 的计算过程如下：

**第一步：计算均方根（RMS）**

$$
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}
$$

**第二步：归一化**

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(x) + \epsilon}
$$

其中 $\epsilon$ 是一个很小的常数（如 $10^{-6}$），用于防止除零。

**第三步：仿射变换（输出）**

$$
y_i = \gamma_i \hat{x}_i
$$

其中 $\gamma$ 是可学习的缩放参数（scale parameter）。

> **注意**：与 LayerNorm 不同，RMSNorm **没有偏置项** $\beta$（bias），也**不需要计算均值**。

## 与 LayerNorm 的对比

![alt text](/img/in-post/post-rms-normalization/image.png)

### LayerNorm 的计算过程

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

$$
\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}
$$

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sigma + \epsilon} + \beta_i
$$

### RMSNorm 的计算过程

$$
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}
$$

$$
y_i = \gamma_i \cdot \frac{x_i}{\text{RMS}(x) + \epsilon}
$$

### 关键区别

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 减去均值 | ✅ 是 | ❌ 否 |
| 除以标准差 | ✅ 是 | ❌ 否（用 RMS） |
| 偏置项 $\beta$ | ✅ 有 | ❌ 无 |
| 可学习参数 | $\gamma, \beta$ | 仅 $\gamma$ |
| 计算速度 | 基线 | **快 ~7-10%** |

RMSNorm 减少了均值计算和偏置项，在实际训练中可以获得约 **7-10%** 的速度提升，同时保持与 LayerNorm 相当的模型性能。

## 代码实现

### 手动实现（简洁版）

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
```

### 手动实现（优化版）

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 rsqrt 避免先 sqrt 再除法，减少一次运算
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

`torch.rsqrt` 直接计算 $\frac{1}{\sqrt{x}}$，比先 `sqrt` 再除法更高效。

### 使用 PyTorch 内置 RMSNorm（PyTorch 2.4+）

```python
# PyTorch 2.4+ 提供了内置的 RMSNorm
rms_norm = nn.RMSNorm(normalized_shape=768, eps=1e-6)

x = torch.randn(2, 10, 768)
output = rms_norm(x)
print(output.shape)  # torch.Size([2, 10, 768])
```

## 在大模型中的应用

现代大语言模型普遍采用 **Pre-Norm** 架构，即在 Attention 和 FFN **之前**进行归一化，而非之后。RMSNorm 是 Pre-Norm 架构中最常用的归一化方式。

以 LLaMA 为例，其 Transformer Block 的典型结构如下：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=eps)
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.attention = MultiHeadAttention(dim, n_heads)
        self.ffn = FeedForward(dim, ffn_dim)

    def forward(self, x):
        # Pre-Norm: 先归一化，再做 Attention，再残差连接
        h = x + self.attention(self.attention_norm(x))
        # Pre-Norm: 先归一化，再做 FFN，再残差连接
        out = h + self.ffn(self.ffn_norm(h))
        return out
```

这种 **Pre-RMSNorm + 残差连接** 的模式已经成为现代 LLM 的标准设计。

## 总结

- RMSNorm 是 LayerNorm 的简化版本，去掉了均值计算和偏置项
- 仅依赖均方根（RMS）进行归一化，计算更高效（快 ~7-10%）
- 在实际应用中**不损失模型精度**
- 已成为 LLaMA、Gemma、Qwen、Mistral 等主流大语言模型的**标准归一化方案**

## 参考

- Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS 2019.
- Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*.
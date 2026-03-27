---
layout: post
title: "理解 <code>LayerNorm</code>"
subtitle: "Layer Normalization 原理与实现"
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

## LayerNorm 介绍

**Layer Normalization** 由 Ba et al. 在 2016 年的论文 *"Layer Normalization"* 中提出。与 Batch Normalization (BatchNorm) 不同，LayerNorm 对**单个样本内的所有特征**进行归一化，而不是在 batch 维度上做统计。

具体来说，对于一个形状为 `(B, H)` 的输入（B 为 batch size，H 为 hidden dimension），BatchNorm 沿着 B 维度计算均值和方差，而 LayerNorm 沿着 H 维度计算。这意味着 LayerNorm 的归一化过程**完全独立于 batch 中的其他样本**。

## 为什么需要 LayerNorm

BatchNorm 在 CNN 中表现优异，但在 RNN 和 Transformer 中存在明显不足：

1. **变长序列问题**：RNN / Transformer 处理的序列长度不一，不同时间步的统计量不稳定，BatchNorm 难以有效计算。
2. **对 batch size 的依赖**：BatchNorm 的均值和方差依赖当前 batch，当 batch size 很小时统计量噪声大，效果显著下降。
3. **推理时的 running statistics**：BatchNorm 需要维护全局的 running mean 和 running variance，在动态场景中可能不准确。

LayerNorm 在**每个样本内部**独立计算统计量，完全不依赖 batch size，因此成为 Transformer 架构的标准选择。

## 数学公式

给定输入向量 $x = (x_1, x_2, \ldots, x_H)$，其中 $H$ 为 hidden dimension 的大小，LayerNorm 的计算过程如下：

**Step 1：计算均值**

$$\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$$

**Step 2：计算方差**

$$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2$$

**Step 3：归一化**

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

其中 $\epsilon$ 是一个很小的常数（默认 $10^{-5}$），防止除零。

**Step 4：仿射变换（Scale & Shift）**

$$y_i = \gamma \hat{x}_i + \beta$$

$\gamma$（scale）和 $\beta$（shift）是可学习参数，形状均为 $(H,)$，初始值分别为 1 和 0。它们的作用是恢复网络的表达能力——如果模型发现归一化后的分布不利于学习，可以通过这两个参数把分布"拉回来"。

## LayerNorm vs BatchNorm

两者的核心区别在于**归一化的维度不同**：

```
输入形状: (B, H)    B = batch size, H = hidden dim

BatchNorm — 沿 batch 维度归一化（每个特征独立）:
            feature_1  feature_2  feature_3
sample_1  [   ↓          ↓          ↓     ]
sample_2  [   ↓          ↓          ↓     ]
sample_3  [   ↓          ↓          ↓     ]
           统计量1     统计量2     统计量3

LayerNorm — 沿 feature 维度归一化（每个样本独立）:
            feature_1  feature_2  feature_3
sample_1  [   →→→→→→→→→→→→→→→→→→→→→→→   ] → 统计量1
sample_2  [   →→→→→→→→→→→→→→→→→→→→→→→   ] → 统计量2
sample_3  [   →→→→→→→→→→→→→→→→→→→→→→→   ] → 统计量3
```

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | batch 维度 | feature 维度 |
| 依赖 batch | 是 | 否 |
| 训练/推理行为 | 不同（running stats） | 相同 |
| 适用场景 | CNN | RNN / Transformer |

## 代码实现

### 手动实现 LayerNorm

```python
import torch
import torch.nn as nn

class ManualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # x: (batch_size, ..., normalized_shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 使用 PyTorch 内置 nn.LayerNorm

```python
import torch
import torch.nn as nn

# 创建输入: batch_size=2, seq_len=4, hidden_dim=8
x = torch.randn(2, 4, 8)

# 对最后一个维度 (hidden_dim=8) 做 LayerNorm
layer_norm = nn.LayerNorm(normalized_shape=8)
output = layer_norm(x)

print(f"输入形状: {x.shape}")          # (2, 4, 8)
print(f"输出形状: {output.shape}")      # (2, 4, 8)
print(f"gamma 形状: {layer_norm.weight.shape}")  # (8,)
print(f"beta 形状: {layer_norm.bias.shape}")     # (8,)
```

### 验证手动实现与 PyTorch 一致

```python
torch.manual_seed(42)
x = torch.randn(2, 4, 8)

manual_ln = ManualLayerNorm(8)
pytorch_ln = nn.LayerNorm(8)

# 使用相同参数
with torch.no_grad():
    manual_ln.gamma.copy_(pytorch_ln.weight)
    manual_ln.beta.copy_(pytorch_ln.bias)

out_manual = manual_ln(x)
out_pytorch = pytorch_ln(x)

print(f"最大差异: {(out_manual - out_pytorch).abs().max().item():.2e}")
# 输出接近 0，验证实现正确
```

## 在 Transformer 中的应用

Transformer 中 LayerNorm 有两种常见的放置方式：

### Post-LN（原始 Transformer）

原始 *"Attention Is All You Need"* 论文中的方式，LayerNorm 放在残差连接**之后**：

```python
class PostLNTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-LN: Add & Norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)       # LN 在残差之后
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)         # LN 在残差之后
        return x
```

### Pre-LN（更稳定的训练）

Pre-LN 将 LayerNorm 放在子层**之前**，梯度流更平滑，训练更稳定，被 GPT-2、BERT 等广泛采用：

```python
class PreLNTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LN: Norm & Add
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out                    # LN 在子层之前
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out                     # LN 在子层之前
        return x
```

**Pre-LN vs Post-LN**：实验表明 Pre-LN 在深层网络中对学习率更鲁棒，不需要 warm-up 也能稳定训练。Post-LN 理论上在收敛后性能略优，但训练更敏感。

## 总结

- LayerNorm 在**单个样本的 feature 维度**上归一化，不依赖 batch 中的其他样本。
- 解决了 BatchNorm 在变长序列和小 batch 场景下的问题。
- 训练和推理行为一致，无需维护 running statistics。
- 是 Transformer 架构的核心组件，Pre-LN 变体提供了更稳定的训练。
- 公式简洁：计算均值 → 方差 → 归一化 → 仿射变换，可学习参数为 $\gamma$ 和 $\beta$。
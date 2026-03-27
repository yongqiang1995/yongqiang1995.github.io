---
layout: post
title: "理解 <code>Batch Normalization</code>"
subtitle: "深度学习中的 Batch Normalization 原理与实践"
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

## Batch Normalization 介绍

Batch Normalization (BN) 由 Sergey Ioffe 和 Christian Szegedy 在 2015 年的论文 *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"* 中提出。它是现代深度学习中最重要的技术之一。

### 为什么需要 Batch Normalization？

在深度网络的训练过程中，每一层的输入分布会随着前面层参数的更新而不断变化，这种现象被称为 **Internal Covariate Shift**。具体来说：

- 网络中每一层的参数在每次迭代后都会更新
- 前面层参数的微小变化会在后续层中被逐层放大
- 后面的层需要不断地适应新的输入分布，导致训练不稳定

这带来的直接后果是：**训练需要更小的学习率、更谨慎的参数初始化**，整体收敛速度很慢。

Batch Normalization 的核心思想是：**在每一层的激活函数之前（或之后），对输入进行标准化处理**，使其均值为 0、方差为 1，从而稳定各层输入的分布。

## 数学逻辑

### 前向传播（Training）

给定一个 mini-batch $B = \{x_1, x_2, \dots, x_m\}$，BN 的计算过程如下：

**Step 1：计算 batch 均值**

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Step 2：计算 batch 方差**

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Step 3：标准化**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

其中 $\epsilon$ 是一个很小的常数（如 $10^{-5}$），用于防止除零。

**Step 4：缩放和平移（Scale and Shift）**

$$y_i = \gamma \hat{x}_i + \beta$$

这里 $\gamma$（scale）和 $\beta$（shift）是**可学习参数**。为什么需要它们？如果只做标准化，网络的表达能力会受限——例如 Sigmoid 激活函数的输入被限制在线性区域附近。引入 $\gamma$ 和 $\beta$ 后，网络可以学习到"恢复"原始分布的能力，即当 $\gamma = \sqrt{\sigma_B^2}$，$\beta = \mu_B$ 时，BN 层等价于恒等变换。

### 推理阶段（Inference）

推理时没有 mini-batch，因此不能直接计算 batch 统计量。BN 使用训练过程中记录的 **running mean** 和 **running variance**：

$$\mu_{\text{running}} \leftarrow (1 - \alpha) \cdot \mu_{\text{running}} + \alpha \cdot \mu_B$$

$$\sigma^2_{\text{running}} \leftarrow (1 - \alpha) \cdot \sigma^2_{\text{running}} + \alpha \cdot \sigma^2_B$$

其中 $\alpha$ 是动量系数（PyTorch 中默认为 0.1）。推理时直接使用这些全局统计量进行标准化。

> **关键点**：训练时必须调用 `model.train()`，推理时必须调用 `model.eval()`，否则 BN 层的行为会不正确。

## 代码实现

### 手动实现 Batch Normalization

```python
import torch
import torch.nn as nn


class ManualBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # running statistics（不参与梯度计算）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # 更新 running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# 验证：与 PyTorch 官方实现对比
torch.manual_seed(42)
x = torch.randn(8, 16)  # batch_size=8, features=16

bn_official = nn.BatchNorm1d(16, momentum=0.1)
bn_manual = ManualBatchNorm1d(16, momentum=0.1)

# 确保参数一致
bn_manual.gamma.data = bn_official.weight.data.clone()
bn_manual.beta.data = bn_official.bias.data.clone()

out_official = bn_official(x)
out_manual = bn_manual(x)
print(f"最大误差: {(out_official - out_manual).abs().max().item():.2e}")
# 输出接近 0，说明实现正确
```

### 在 CNN 中使用 BatchNorm2d

对于卷积网络，BN 在 $(N, H, W)$ 维度上计算统计量，每个 channel 独立归一化：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 对 32 个 channel 分别做 BN
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        # Conv -> BN -> ReLU 是最常见的顺序
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = SimpleCNN()
dummy_input = torch.randn(4, 3, 32, 32)  # batch=4, 3通道, 32x32
output = model(dummy_input)
print(f"输出维度: {output.shape}")  # torch.Size([4, 10])

# 查看 BN 层的参数
print(f"BN1 weight (gamma) shape: {model.bn1.weight.shape}")
print(f"BN1 running_mean shape: {model.bn1.running_mean.shape}")
```

## 优势与局限

### 优势

| 优势 | 说明 |
|------|------|
| **加速收敛** | 允许使用更大的学习率，减少训练所需的迭代次数 |
| **正则化效果** | 每个 mini-batch 的统计量引入了噪声，起到类似 Dropout 的正则化效果 |
| **降低初始化敏感性** | 对权重初始化的依赖降低，网络训练更加稳定 |
| **缓解梯度问题** | 标准化后梯度的尺度更加稳定，减轻梯度消失/爆炸 |

### 局限

- **对 batch size 敏感**：当 batch size 很小时，batch 统计量的估计不准确，性能显著下降
- **不适用于变长序列**：在 RNN/Transformer 等处理变长输入的场景中效果不佳
- **训练和推理行为不一致**：需要小心管理 `train()`/`eval()` 模式切换

## 与其他 Normalization 方法的对比

不同的 Normalization 方法在**计算统计量的维度**上有本质区别。假设输入 tensor 的形状为 $(N, C, H, W)$：

| 方法 | 归一化维度 | 适用场景 |
|------|-----------|---------|
| **BatchNorm** | $(N, H, W)$，沿 batch 和空间维度 | CNN，大 batch size |
| **LayerNorm** | $(C, H, W)$，沿 channel 和空间维度 | Transformer，NLP 任务 |
| **InstanceNorm** | $(H, W)$，仅空间维度 | 风格迁移，图像生成 |
| **GroupNorm** | $(C/G, H, W)$，将 channel 分组 | 小 batch size 的视觉任务 |

简单来说：**BatchNorm 依赖 batch 维度，LayerNorm 依赖 feature 维度**。这就是为什么 Transformer 架构普遍使用 LayerNorm——self-attention 的 batch size 通常较小，且序列长度可变。

## 总结

Batch Normalization 的核心要点：

1. **通过标准化层输入分布**来解决 Internal Covariate Shift，加速训练
2. **引入可学习参数** $\gamma$ 和 $\beta$，保证网络的表达能力不受损
3. **训练时使用 batch 统计量，推理时使用 running statistics**，两者行为不同
4. 在 CNN 中效果显著，但在 **小 batch size 或序列模型**中建议使用 LayerNorm 或 GroupNorm

理解 BN 是理解现代深度学习架构的基础——无论是 ResNet 中的标配 BN，还是 Transformer 中替代它的 LayerNorm，核心思想都是**通过控制中间层的分布来稳定和加速训练**。
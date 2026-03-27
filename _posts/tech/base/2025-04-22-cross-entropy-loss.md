---
layout: post
title: "交叉熵损失函数详解"
subtitle: "Cross Entropy Loss 原理、推导与实现"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
mathjax: true
tags:
  - 损失函数
  - 深度学习基础
---

## 1. 交叉熵简介

交叉熵 (Cross Entropy) 源自信息论，要理解它需要先了解几个基本概念。

### 1.1 信息量与熵

**信息量**衡量一个事件发生时带来的"惊讶程度"。事件 $x$ 的信息量定义为：

$$
I(x) = -\log p(x)
$$

**香农熵 (Shannon Entropy)** 是信息量的期望值，衡量随机变量的不确定性：

$$
H(p) = -\sum_{x} p(x) \log p(x)
$$

熵越大，分布越均匀，不确定性越高；熵越小，分布越集中，不确定性越低。

### 1.2 交叉熵

交叉熵衡量用分布 $q$ 编码来自分布 $p$ 的数据时所需的平均比特数：

$$
H(p, q) = -\sum_{x} p(x) \log q(x)
$$

当 $q = p$ 时，$H(p, q) = H(p)$，即交叉熵等于熵本身，此时编码效率最高。

### 1.3 KL 散度

KL 散度 (Kullback-Leibler Divergence) 衡量两个分布之间的差异：

$$
D_{KL}(p \| q) = H(p, q) - H(p) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

由于真实分布 $p$ 的熵 $H(p)$ 是常数，**最小化交叉熵等价于最小化 KL 散度**，这就是交叉熵能作为损失函数的理论基础。

## 2. 交叉熵损失函数

### 2.1 二分类交叉熵 (Binary Cross Entropy)

对于二分类问题，真实标签 $y \in \{0, 1\}$，模型预测概率 $\hat{y} \in (0, 1)$：

$$
L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

- 当 $y = 1$ 时，$L = -\log(\hat{y})$，预测越接近 1 损失越小
- 当 $y = 0$ 时，$L = -\log(1 - \hat{y})$，预测越接近 0 损失越小

### 2.2 多分类交叉熵 (Categorical Cross Entropy)

对于 $C$ 个类别的分类问题，真实标签为 one-hot 向量 $\mathbf{y}$，模型输出概率分布 $\hat{\mathbf{y}}$：

$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

由于 one-hot 编码中只有真实类别 $k$ 对应的 $y_k = 1$，其余为 0，上式简化为：

$$
L = -\log(\hat{y}_k)
$$

这意味着损失仅取决于模型对正确类别的预测概率。

## 3. 与 Softmax 的关系

### 3.1 Softmax 函数

Softmax 将模型输出的原始分数 (logits) 转换为概率分布：

$$
\sigma(\vec{z})_{i} = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

满足 $\sum_i \sigma(\vec{z})_i = 1$ 且 $\sigma(\vec{z})_i > 0$。

### 3.2 Softmax 的导数

将 Softmax 函数记为 $S_i$，则导数为：

$$
\frac{\partial S_i}{\partial z_j} =
\begin{cases}
S_i(1 - S_j) & i = j \\
-S_i S_j     & i \neq j
\end{cases}
$$

### 3.3 组合梯度推导

将 Softmax 输出代入交叉熵损失 $L = -\log(S_k)$，对 logit $z_j$ 求导：

$$
\frac{\partial L}{\partial z_j} = -\frac{1}{S_k} \cdot \frac{\partial S_k}{\partial z_j}
$$

- 当 $j = k$ 时：$\frac{\partial L}{\partial z_k} = -\frac{1}{S_k} \cdot S_k(1 - S_k) = S_k - 1 = \hat{y}_k - 1$
- 当 $j \neq k$ 时：$\frac{\partial L}{\partial z_j} = -\frac{1}{S_k} \cdot (-S_k S_j) = S_j = \hat{y}_j$

统一写成向量形式：

$$
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}
$$

这个结果非常优雅——**Softmax + 交叉熵的梯度就是预测概率与真实标签之间的差值**。这也是为什么二者总是组合使用的原因。

## 4. 数值稳定性

### 4.1 溢出问题

直接计算 $e^{z_i}$ 容易导致数值上溢。解决方法是 **Log-Sum-Exp** 技巧，利用 Softmax 的平移不变性：

$$
\log \sum_{j} e^{z_j} = m + \log \sum_{j} e^{z_j - m}, \quad m = \max_j z_j
$$

减去最大值后指数运算不会溢出，同时结果保持不变。

### 4.2 PyTorch 的设计选择

PyTorch 中 `nn.CrossEntropyLoss` 接收**原始 logits**（而非 Softmax 输出），内部将 Softmax 和交叉熵合并计算。这样做有两个好处：

1. **数值稳定**：利用 log-sum-exp 技巧避免溢出
2. **计算高效**：避免先算 Softmax 再取 log 的冗余操作

## 5. 代码实现

### 5.1 手动实现

```python
import torch
import torch.nn.functional as F

def cross_entropy_manual(logits, targets):
    """手动实现交叉熵损失（含数值稳定处理）"""
    # Log-Sum-Exp 技巧
    max_val = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = max_val + torch.log(torch.exp(logits - max_val).sum(dim=-1, keepdim=True))
    log_probs = logits - log_sum_exp
    # 取出真实类别对应的 log 概率
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()

# 测试
logits = torch.randn(4, 5)       # batch_size=4, num_classes=5
targets = torch.randint(0, 5, (4,))
print(f"手动实现: {cross_entropy_manual(logits, targets):.4f}")
print(f"PyTorch:  {F.cross_entropy(logits, targets):.4f}")
```

### 5.2 PyTorch 内置接口

```python
import torch
import torch.nn as nn

logits = torch.randn(4, 5)
targets = torch.randint(0, 5, (4,))

# 方式一：nn.CrossEntropyLoss（输入为原始 logits）
criterion = nn.CrossEntropyLoss()
loss1 = criterion(logits, targets)

# 方式二：nn.NLLLoss（输入为 log-softmax 输出）
log_probs = torch.log_softmax(logits, dim=-1)
loss2 = nn.NLLLoss()(log_probs, targets)

print(f"CrossEntropyLoss: {loss1:.4f}")
print(f"NLLLoss:          {loss2:.4f}")  # 结果一致
```

### 5.3 常见陷阱：双重 Softmax

```python
# ❌ 错误用法：先 softmax 再传给 CrossEntropyLoss
probs = torch.softmax(logits, dim=-1)
wrong_loss = nn.CrossEntropyLoss()(probs, targets)  # 内部会再做一次 softmax！

# ✅ 正确用法：直接传入 logits
correct_loss = nn.CrossEntropyLoss()(logits, targets)
```

`CrossEntropyLoss` 内部已经包含 Softmax，如果提前对 logits 做了 Softmax，相当于做了两次，会导致梯度信号被压缩，训练效果变差。

## 6. 总结

| 概念 | 说明 |
|------|------|
| 交叉熵 | 衡量预测分布与真实分布的差异 |
| 二分类 CE | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| 多分类 CE | $-\sum y_c \log \hat{y}_c$，one-hot 下简化为 $-\log \hat{y}_k$ |
| Softmax + CE 梯度 | $\hat{\mathbf{y}} - \mathbf{y}$，形式简洁，便于反向传播 |
| 数值稳定性 | 使用 log-sum-exp 技巧，PyTorch 接收 logits 而非概率值 |
| 常见错误 | 不要在 `CrossEntropyLoss` 之前手动调用 Softmax |


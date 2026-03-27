---
layout: post
title: "理解 <code>Jacobian</code> 矩阵"
subtitle: "从数学定义到深度学习中的反向传播"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
mathjax: true
tags:
  - 数学基础
  - 深度学习基础
---

## Jacobian 矩阵介绍

在多元微积分中，**Jacobian 矩阵**（雅可比矩阵）是一个非常重要的概念。它以德国数学家 **Carl Gustav Jacob Jacobi** 的名字命名，描述了一个向量值函数的所有一阶偏导数。

对于一个向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，即输入为 $n$ 维向量、输出为 $m$ 维向量的函数：

$$
\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \dots, x_n) \\ f_2(x_1, \dots, x_n) \\ \vdots \\ f_m(x_1, \dots, x_n) \end{pmatrix}
$$

其 Jacobian 矩阵就是一个 $m \times n$ 的矩阵，包含了每个输出分量对每个输入分量的偏导数。简单来说，Jacobian 矩阵是标量函数梯度概念在向量值函数上的自然推广。

## 数学定义

Jacobian 矩阵 $\mathbf{J} \in \mathbb{R}^{m \times n}$ 的定义如下：

$$
\mathbf{J} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}
$$

其中每个元素为：

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

即矩阵的第 $i$ 行、第 $j$ 列元素是第 $i$ 个输出对第 $j$ 个输入的偏导数。

### 一个简单的例子

设函数 $\mathbf{f}(x_1, x_2) = (f_1, f_2)$，其中：

$$
f_1 = x_1^2 + x_2, \quad f_2 = x_1 x_2
$$

分别计算各偏导数：

$$
\frac{\partial f_1}{\partial x_1} = 2x_1, \quad \frac{\partial f_1}{\partial x_2} = 1, \quad \frac{\partial f_2}{\partial x_1} = x_2, \quad \frac{\partial f_2}{\partial x_2} = x_1
$$

因此，Jacobian 矩阵为：

$$
\mathbf{J} = \begin{pmatrix} 2x_1 & 1 \\ x_2 & x_1 \end{pmatrix}
$$

在点 $(x_1, x_2) = (1, 2)$ 处，$\mathbf{J} = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}$。

## 几何意义

Jacobian 矩阵的几何意义可以从两个角度来理解：

**线性近似**：Jacobian 矩阵表示向量值函数在某一点附近的**最佳线性近似**。具体地说，在 $\mathbf{x}_0$ 附近有：

$$
\mathbf{f}(\mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + \mathbf{J}(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)
$$

这与标量函数中 $f(x) \approx f(x_0) + f'(x_0)(x - x_0)$ 的思想完全一致。

**体积缩放**：当 $m = n$ 时（即方阵情形），Jacobian 行列式 $\det(\mathbf{J})$ 衡量了函数在局部对体积的缩放程度。如果 $\lvert\det(\mathbf{J})\rvert > 1$，表示局部体积被放大；如果 $\lvert\det(\mathbf{J})\rvert < 1$，表示局部体积被缩小。这个性质在概率密度函数的变量替换中至关重要。

## 在深度学习中的应用

Jacobian 矩阵在深度学习中扮演着核心角色，尤其体现在**反向传播**（Backpropagation）算法中。

### 反向传播与 Jacobian 向量积

神经网络可以看作一系列函数的复合。考虑某一层的映射 $\mathbf{y} = f(\mathbf{x})$，其中 $\mathbf{x}$ 是该层的输入，$\mathbf{y}$ 是输出。在反向传播过程中，假设我们已经知道损失函数 $L$ 对 $\mathbf{y}$ 的梯度 $\frac{\partial L}{\partial \mathbf{y}}$，需要计算 $L$ 对 $\mathbf{x}$ 的梯度。

根据链式法则：

$$
\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}^T \frac{\partial L}{\partial \mathbf{y}}
$$

其中 $\mathbf{J}$ 是 $f$ 在 $\mathbf{x}$ 处的 Jacobian 矩阵。这个运算被称为 **VJP**（Vector-Jacobian Product，向量-雅可比积）。本质上，反向传播就是从输出层到输入层，逐层计算 VJP 的过程。

### Softmax 函数的 Jacobian

Softmax 函数是深度学习中常见的激活函数，其 Jacobian 矩阵尤其经典。设 Softmax 输出为 $\sigma_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$，其 Jacobian 矩阵为：

$$
\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)
$$

其中 $\delta_{ij}$ 是 Kronecker delta 符号（当 $i = j$ 时为 1，否则为 0）。用矩阵形式可以写作：

$$
\mathbf{J}_{\text{softmax}} = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T
$$

> 关于 Softmax 函数的详细推导，可以参考本站的 Softmax 相关文章。

### 梯度消失与梯度爆炸

在深度网络中，反向传播时需要连续乘以多层的 Jacobian 矩阵。如果 Jacobian 矩阵的谱范数（最大奇异值）持续大于 1，梯度会指数增长（**梯度爆炸**）；如果持续小于 1，梯度会指数衰减（**梯度消失**）。这也是 ResNet 残差连接、梯度裁剪等技术出现的重要原因。

## 代码实现

PyTorch 提供了 `torch.autograd.functional.jacobian` 来直接计算 Jacobian 矩阵。下面以前文的例子进行演示：

```python
import torch
from torch.autograd.functional import jacobian


def func(x):
    """f(x1, x2) = (x1^2 + x2, x1 * x2)"""
    return torch.stack([x[0] ** 2 + x[1], x[0] * x[1]])


# 在点 (1.0, 2.0) 处计算 Jacobian
x = torch.tensor([1.0, 2.0])
J = jacobian(func, x)
print("Jacobian 矩阵:")
print(J)
# 输出:
# tensor([[2., 1.],
#         [2., 1.]])
```

对于更复杂的神经网络层，也可以用同样的方式计算：

```python
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


def softmax_func(z):
    return torch.softmax(z, dim=0)


z = torch.tensor([1.0, 2.0, 3.0])
J_softmax = jacobian(softmax_func, z)
print("Softmax Jacobian 矩阵:")
print(J_softmax)

# 验证: diag(sigma) - sigma @ sigma^T
sigma = torch.softmax(z, dim=0)
J_manual = torch.diag(sigma) - sigma.unsqueeze(1) @ sigma.unsqueeze(0)
print("手动计算的 Jacobian 矩阵:")
print(J_manual)
print("两者是否一致:", torch.allclose(J_softmax, J_manual))
```

## 总结

- **Jacobian 矩阵**是向量值函数所有一阶偏导数组成的 $m \times n$ 矩阵，是梯度在向量函数上的自然推广。
- 它表示函数在某一点的**最佳线性近似**，其行列式度量局部的体积缩放。
- 深度学习中的**反向传播本质上就是逐层计算 VJP**（向量-雅可比积），即 $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}^T \frac{\partial L}{\partial \mathbf{y}}$。
- Jacobian 矩阵的谱特性直接决定了梯度的传播行为，与**梯度消失/爆炸**问题密切相关。
- PyTorch 的 `torch.autograd.functional.jacobian` 提供了便捷的 Jacobian 计算接口。
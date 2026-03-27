---
layout: post
title: "最小二乘法详解"
subtitle: "Least Square Method 从原理到实现"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
mathjax: true
tags:
  - 数学基础
  - 机器学习
---

## 最小二乘法简介

**最小二乘法**（Least Square Method）是数学和统计学中最基本的参数估计方法之一。其核心思想是：通过最小化残差的平方和，找到最优的模型参数，使得模型的预测值尽可能接近真实观测值。

最小二乘法最早由法国数学家 Legendre 于 1805 年提出，随后 Gauss 独立发展了该方法并给出了更完善的理论基础。时至今日，最小二乘法仍然是统计学、信号处理和机器学习中最常用的优化技术之一，也是理解线性回归、曲线拟合等核心算法的基础。

## 问题定义

假设我们有 $n$ 个数据点 $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$，希望找到一条直线：

$$\hat{y} = wx + b$$

使得所有数据点到该直线的**残差平方和**最小：

$$S(w, b) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - wx_i - b)^2$$

其中 $w$ 为斜率，$b$ 为截距，$y_i - \hat{y}_i$ 称为第 $i$ 个观测的**残差**（residual）。

## 数学推导

### 定义损失函数

将问题形式化为求解以下损失函数的最小值：

$$L(w, b) = \sum_{i=1}^{n} (y_i - wx_i - b)^2$$

### 求偏导数并令其为零

对 $w$ 求偏导：

$$\frac{\partial L}{\partial w} = -2\sum_{i=1}^{n} x_i(y_i - wx_i - b) = 0$$

对 $b$ 求偏导：

$$\frac{\partial L}{\partial b} = -2\sum_{i=1}^{n} (y_i - wx_i - b) = 0$$

### 求解闭式解

由第二个方程可得：

$$\sum_{i=1}^{n} y_i = w\sum_{i=1}^{n} x_i + nb$$

即 $b = \bar{y} - w\bar{x}$，其中 $\bar{x}$、$\bar{y}$ 分别为 $x$ 和 $y$ 的均值。

将 $b$ 代入第一个方程，整理可得斜率 $w$ 的闭式解：

$$w = \frac{n\sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n\sum_{i=1}^{n} x_i^2 - \left(\sum_{i=1}^{n} x_i\right)^2}$$

### 矩阵形式

当问题推广到多元线性回归时，模型可以写成矩阵形式：

$$y = X\theta + \epsilon$$

其中 $X$ 为设计矩阵（每行一个样本，包含截距项对应的 1），$\theta$ 为参数向量。最小二乘的目标变为最小化 $\|y - X\theta\|^2$，对 $\theta$ 求导并令其为零，可得到著名的**正规方程**（Normal Equation）：

$$\hat{\theta} = (X^TX)^{-1}X^Ty$$

这就是最小二乘法的矩阵闭式解。使用该公式的前提是 $X^TX$ 可逆，否则需要引入正则化（如岭回归）或使用伪逆。

## 代码实现

### 手动实现

```python
import numpy as np

# 生成示例数据
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2.5 * x + 1.3 + np.random.randn(50) * 2

# 手动计算最小二乘解
n = len(x)
w = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
    (n * np.sum(x ** 2) - np.sum(x) ** 2)
b = np.mean(y) - w * np.mean(x)
print(f"手动实现: w = {w:.4f}, b = {b:.4f}")
```

### 使用矩阵形式

```python
# 构建设计矩阵 [x, 1]
X = np.column_stack([x, np.ones(n)])

# 正规方程求解
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"矩阵形式: w = {theta[0]:.4f}, b = {theta[1]:.4f}")
```

### 使用 NumPy 内置函数

```python
# numpy.linalg.lstsq（推荐，数值更稳定）
theta_np, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(f"np.lstsq:  w = {theta_np[0]:.4f}, b = {theta_np[1]:.4f}")
```

### 使用 scikit-learn

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
print(f"sklearn:   w = {model.coef_[0]:.4f}, b = {model.intercept_:.4f}")
```

以上四种方式的结果应完全一致，验证了我们推导的正确性。

## 与梯度下降法的对比

| 特性 | 最小二乘法（闭式解） | 梯度下降法 |
|------|---------------------|-----------|
| 求解方式 | 直接计算解析解 | 迭代逼近最优解 |
| 时间复杂度 | $O(n \cdot d^2 + d^3)$ | $O(k \cdot n \cdot d)$，$k$ 为迭代次数 |
| 适用规模 | 特征维度 $d$ 较小时高效 | 大规模数据和高维特征 |
| 数值稳定性 | 矩阵求逆可能不稳定 | 学习率选择不当可能不收敛 |
| 超参数 | 无 | 需要调节学习率、迭代次数 |

**选择建议**：

- 当特征维度 $d < 10^4$ 且数据量适中时，优先使用闭式解，计算速度快且无需调参。
- 当特征维度很高或数据量极大时，梯度下降法（尤其是小批量随机梯度下降 SGD）更加高效。
- 在实际工程中，`numpy.linalg.lstsq` 使用 SVD 分解来求解，兼顾了数值稳定性和效率。

## 在深度学习中的联系

最小二乘法与深度学习有着密切的联系：

**MSE 损失即最小二乘原理**：深度学习中广泛使用的均方误差损失（Mean Squared Error）本质上就是最小二乘原理的直接应用：

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

最小化 MSE 等价于最小二乘法的目标函数（仅差一个常数因子 $\frac{1}{n}$）。

**线性回归 = 最简单的神经网络**：一个没有激活函数、只有一个神经元的单层网络，其前向传播为 $\hat{y} = wx + b$，使用 MSE 损失训练，这与最小二乘线性回归完全等价。可以说，线性回归是神经网络的起点。

从这个视角来看，深度学习中的训练过程可以理解为：在更复杂的非线性模型上，使用梯度下降法求解一个广义的"最小二乘"问题。

## 总结

- 最小二乘法通过最小化残差平方和来估计模型参数，是回归分析的基石。
- 对于线性模型，可以推导出闭式解（正规方程），也可以使用矩阵形式统一表达。
- 实际应用中可以选择手动实现、NumPy 或 scikit-learn 等工具，结果一致。
- 与梯度下降法相比，闭式解适合中小规模问题，梯度下降适合大规模场景。
- 最小二乘法的思想贯穿了从传统统计到现代深度学习的整个体系。
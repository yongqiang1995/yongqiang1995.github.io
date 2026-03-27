import numpy as np

# def softmax(z):
#     # z is a input vector
#     exp_zi = [np.exp(z_i) for z_i in z]
#     exp_sum = np.array([np.exp(z_j) for z_j in z.flatten()]).sum()
#     softmax_o = exp_zi / exp_sum
#     return softmax_o

import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

# vector_z = np.array([1, 2, 3, 4, 5])
vector_z = np.array([2, 1, 0.1])
print(f"softmax(vector_z)=", softmax(vector_z))
exit()

# 输出
# softmax(vector_z)= [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]

import torch
import torch.nn.functional as F

vector_z = torch.from_numpy(vector_z).to(dtype=torch.float32)
print(f"softmax(vector_z)=", F.softmax(vector_z, dim=0))

# 输出
# softmax(vector_z)= tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])

# 绘制 softmax-1d 函数
import matplotlib.pyplot as plt

# 生成输入数据（1D）
# import pdb; pdb.set_trace()
x = np.linspace(-50, 50, 1000)
probabilities = softmax(x)

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制曲线
ax.plot(x, probabilities, 
        color='#2E86C1', 
        linewidth=2.5, 
        label='Softmax')

# 可视化优化设置
ax.spines['top'].set_visible(False)    # 隐藏顶部边框[1](@ref)
ax.spines['right'].set_visible(False)   # 隐藏右侧边框
ax.spines['left'].set_position(('axes', 0.05))  # 移动左侧坐标轴
ax.spines['bottom'].set_position(('axes', 0.1)) # 移动底部坐标轴

# 添加标签和标题
ax.set_xlabel('Input Value', fontsize=12, labelpad=10)
ax.set_ylabel('Probability', fontsize=12, labelpad=10)
ax.set_title('1D Softmax Function (Numerically Stable)', 
            fontsize=14, pad=20)

# 设置网格和刻度
ax.grid(True, 
       linestyle='--', 
       alpha=0.7,
       color='#D5D8DC')
ax.tick_params(axis='both', 
              which='major', 
              labelsize=10)

# 高亮关键区域
ax.fill_between(x, probabilities, 
               where=(x > 5), 
               color='#F1948A', 
               alpha=0.3, 
               label='High Confidence')
ax.fill_between(x, probabilities, 
               where=(x < -5), 
               color='#7DCEA0', 
               alpha=0.3, 
               label='Low Confidence')

plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()


#####

import numpy as np
import matplotlib.pyplot as plt

def stable_softmax_2d(x1):
    """模拟二维输入的Softmax（第二个输入固定为0）"""
    x = np.column_stack([x1, np.zeros_like(x1)])  # 构建二维输入 [x1, 0]
    x = x - np.max(x, axis=1, keepdims=True)      # 数值稳定处理
    exp_x = np.exp(x)
    import pdb; pdb.set_trace()
    return exp_x[:, 0] / np.sum(exp_x, axis=1)    # 返回第一个元素的概率

# 生成输入数据（与参考图一致的范围）
x1 = np.linspace(-50, 50, 1000)  # 关键：大范围输入以展示饱和特性
softmax = stable_softmax_2d(x1)
derivative = softmax * (1 - softmax)  # 导数公式

# 创建专业可视化
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x1, softmax, 
         color="#3498DB", 
         linewidth=3, 
         label=r"$\sigma_{1D}$ (Softmax)")
plt.plot(x1, derivative, 
         color="#E74C3C", 
         linewidth=3, 
         linestyle="--", 
         label=r"$\frac{d\sigma_{1D}}{dx_1}$")

# 坐标轴和样式设置
plt.xlim(-50, 50)
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(-50, 60, 10), fontsize=12)
plt.yticks(np.linspace(0, 1, 5), fontsize=12)
plt.xlabel(r"$x_1$", fontsize=14, labelpad=10)
plt.ylabel("Probability", fontsize=14, labelpad=10)

# 网格和背景优化
plt.grid(True, linestyle="--", alpha=0.7, color="#BDC3C7")
plt.gca().set_facecolor("#F4F6F6")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# 图例和标题
plt.legend(loc="upper left", 
          fontsize=12, 
          framealpha=0.9,
          facecolor="white")
plt.title("1-Dimensional Softmax (σ) and its Derivative", 
         fontsize=16, pad=20)

# 高亮关键区域
plt.fill_between(x1, softmax, 
                where=(x1 > 25), 
                color="#F1948A", alpha=0.1)
plt.fill_between(x1, softmax, 
                where=(x1 < -25), 
                color="#7DCEA0", alpha=0.1)

plt.tight_layout()
plt.show()
"""
手撕激活函数：Sigmoid、Softmax
================================
面试高频手撕题，必须能默写
"""

import torch
import numpy as np


# ============================================================
# 1. Sigmoid
# ============================================================
# 公式：σ(x) = 1 / (1 + e^(-x))
# 作用：把任意实数映射到 (0, 1)，常用于二分类输出层
# 导数：σ'(x) = σ(x) × (1 - σ(x))   ← 记住这个，面试常问

def sigmoid(x):
    """
    数值稳定版本：对于大负数，e^(-x)会溢出
    所以分情况处理：
      x >= 0: 1 / (1 + e^(-x))        ← 正常算
      x < 0:  e^x / (1 + e^x)         ← 等价变换，避免溢出
    """
    result = np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),           # x>=0 时直接算
        np.exp(x) / (1 + np.exp(x))     # x<0 时用等价形式
    )
    return result


def sigmoid_derivative(x):
    """sigmoid的导数：σ(x) × (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


# ============================================================
# 2. Softmax
# ============================================================
# 公式：softmax(xi) = e^xi / Σ e^xj
# 作用：把一组实数映射到 (0, 1) 且和为1，常用于多分类输出层
# 导数：∂softmax_i/∂x_j = softmax_i × (δ_ij - softmax_j)
#       δ_ij = 1 if i==j else 0

def softmax(x):
    """
    数值稳定版本：e^x 会溢出，所以先减去最大值
    softmax(xi - max) = e^(xi-max) / Σ e^(xj-max)
    数学上等价，但数值更稳定（最大值变成 e^0=1，不会溢出）
    """
    # 减去最大值，防止溢出
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_jacobian(x):
    """
    softmax的雅可比矩阵（导数矩阵）
    J[i][j] = ∂softmax_i/∂x_j = softmax_i × (δ_ij - softmax_j)

    用途：反向传播时，loss对x的梯度 = J^T × ∂loss/∂softmax
    """
    s = softmax(x)
    # 构造雅可比矩阵
    n = len(s)
    jacobian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i][j] = s[i] * (1 - s[i])       # 对角线
            else:
                jacobian[i][j] = -s[i] * s[j]             # 非对角线
    return jacobian


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # ---- Sigmoid 测试 ----
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("=== Sigmoid ===")
    print(f"输入: {x}")
    print(f"输出: {sigmoid(x).round(4)}")
    print(f"导数: {sigmoid_derivative(x).round(4)}")
    print(f"验证: sigmoid(0)应该=0.5 → {sigmoid(np.array([0.0]))}")

    # 对比PyTorch
    x_torch = torch.tensor(x, dtype=torch.float32)
    print(f"PyTorch: {torch.sigmoid(x_torch).numpy().round(4)}")

    # ---- Softmax 测试 ----
    print("\n=== Softmax ===")
    x = np.array([2.0, 1.0, 0.5])
    print(f"输入: {x}")
    print(f"输出: {softmax(x).round(4)}")
    print(f"验证: 输出之和应该=1 → {softmax(x).sum():.4f}")

    # 对比PyTorch
    x_torch = torch.tensor(x, dtype=torch.float32)
    print(f"PyTorch: {torch.softmax(x_torch, dim=0).numpy().round(4)}")

    # 雅可比矩阵
    print(f"\n雅可比矩阵:\n{softmax_jacobian(x).round(4)}")

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




"""
手撕损失函数：MSE、Cross Entropy
==================================
面试高频手撕题，必须能默写
"""

import torch
import numpy as np


# ============================================================
# 1. MSE - 均方误差损失
# ============================================================
# 公式：MSE = (1/n) × Σ (y_pred - y_true)²
# 用途：回归任务（预测价格、评分等）
# 导数：∂MSE/∂y_pred = (2/n) × (y_pred - y_true)

def mse_loss(y_pred, y_true):
    """
    手撕MSE
    y_pred: 预测值，任意shape
    y_true: 真实值，和y_pred同shape
    返回: 标量loss
    """
    return np.mean((y_pred - y_true) ** 2)


# ============================================================
# 2. BCE - 二分类交叉熵损失
# ============================================================
# 公式：BCE = -(1/n) × Σ [y×log(p) + (1-y)×log(1-p)]
# 用途：二分类（点击/不点击，是/否）
# 注意：p是sigmoid后的概率值，不是logit
#
# 推导：
#   当y=1时，loss = -log(p)      → p越大loss越小 ✓
#   当y=0时，loss = -log(1-p)    → p越小loss越小 ✓

def bce_loss(y_pred, y_true):
    """
    手撕BCE
    y_pred: 预测概率，经过sigmoid，范围(0,1)
    y_true: 真实标签，0或1
    """
    # 裁剪防止log(0) = -inf
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


# ============================================================
# 3. CE - 多分类交叉熵损失
# ============================================================
# 公式：CE = -(1/n) × Σ Σ y_ij × log(p_ij)
#        y_ij: 第i个样本第j类的标签（one-hot）
#        p_ij: 第i个样本第j类的预测概率（softmax后）
#
# 简化（标签是类别索引时）：
#   CE = -(1/n) × Σ log(p_i_true_class)
#   即：只需要关注正确类别的预测概率
#
# 为什么用CE不用MSE？
#   MSE的梯度在sigmoid饱和区很小 → 训练慢
#   CE的梯度 = p - y → 不会饱和 → 训练快

def cross_entropy_loss(y_pred_logits, y_true):
    """
    手撕多分类CE
    y_pred_logits: 模型原始输出（logits），未经softmax，shape=(n, num_classes)
    y_true: 真实类别索引，shape=(n,)，值为0,1,2,...

    内部会先做softmax，再算CE
    """
    n = y_pred_logits.shape[0]

    # Step 1: 数值稳定的softmax
    # 减去最大值防止溢出
    logits_shifted = y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Step 2: 取出正确类别的概率
    # y_true是类别索引，用它选出对应的概率
    correct_probs = probs[np.arange(n), y_true]

    # Step 3: -log(p) 求平均
    return -np.mean(np.log(correct_probs + 1e-7))

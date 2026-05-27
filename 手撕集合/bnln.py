import numpy as np

def simple_bn(x, gamma=1, beta=0, eps=1e-5):
    """
    最简单的BN实现
    x: 输入数据 [batch, features]
    gamma: 缩放参数
    beta: 偏移参数
    """
    # 沿着batch维度计算均值和方差
    mean = np.mean(x, axis=0)    # 对每个特征，计算所有样本的均值
    var = np.var(x, axis=0)      # 对每个特征，计算所有样本的方差
    
    # 归一化
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # 缩放和偏移
    return gamma * x_norm + beta

# 测试
x = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
print("输入:", x)
print("BN输出:", simple_bn(x))

def simple_ln(x, gamma=1, beta=0, eps=1e-5):
    """
    最简单的LN实现
    x: 输入数据 [batch, features]
    gamma: 缩放参数
    beta: 偏移参数
    """
    # 沿着特征维度计算均值和方差（对每个样本单独计算）
    mean = np.mean(x, axis=1, keepdims=True)    # 对每个样本，计算所有特征的均值
    var = np.var(x, axis=1, keepdims=True)       # 对每个样本，计算所有特征的方差
    
    # 归一化
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # 缩放和偏移
    return gamma * x_norm + beta

# 测试
print("LN输出:", simple_ln(x))
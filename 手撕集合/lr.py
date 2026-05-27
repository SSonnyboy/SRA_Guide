import numpy as np

def linear_regression_gd(X, y, learning_rate=0.01, n_iters=1000):
    """
    使用梯度下降法求解线性回归参数
    X: 输入特征矩阵, shape (m, n_features)
    y: 目标值向量, shape (m, )
    """
    # 初始化参数
    m = len(y)
    # 通常将权重初始化为0或很小的随机数，偏置初始化为0
    w = np.zeros(X.shape[1]) # 假设X已经包含了偏置项对应的常数列1，或者...
    b = 0
    # 更常见的做法是给X添加一列1，然后将w和b合并成一个参数向量theta
    # 这里我们分开写，更清晰

    # 记录损失历史，用于可视化
    cost_history = []

    for i in range(n_iters):
        # 1. 计算当前预测值
        y_pred = np.dot(X, w) + b

        # 2. 计算误差（损失）
        error = y_pred - y
        cost = (1/(2*m)) * np.sum(error**2)
        cost_history.append(cost)

        # 3. 计算梯度（最关键的部分）
        dw = (1/m) * np.dot(X.T, error) # 对w的梯度
        db = (1/m) * np.sum(error)       # 对b的梯度

        # 4. 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, cost_history

# 示例用法
X = np.array([[1, 1], [1, 2], [1, 3]]) # 假设特征
y = np.array([2, 3, 4])
w, b, history = linear_regression_gd(X, y)
print(f"拟合的权重: {w}, 拟合的偏置: {b}")
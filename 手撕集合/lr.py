import numpy as np

# 🔥 完全仿照你之前 TwoLayerDNN 的结构写的线性回归
class LinearRegression:
    def __init__(self, input_dim, learning_rate=0.01):
        """
        初始化线性回归（等价于 0层神经网络）
        格式和 DNN 完全一致
        """
        # 学习率（和DNN一样）
        self.lr = learning_rate

        # 参数初始化（和DNN格式完全一样）
        self.W = np.random.randn(input_dim, 1) * 0.01  # 权重
        self.b = np.zeros((1, 1))                      # 偏置

    def forward(self, X):
        """
        前向传播（和DNN格式完全一样）
        线性回归就是：没有激活函数的前向
        """
        # 只有一行：线性变换
        y_pred = np.dot(X, self.W) + self.b
        return y_pred

    def backward(self, X, Y, y_pred):
        """
        反向传播（和DNN格式、逻辑完全一样！）
        梯度计算 → 参数更新
        """
        m = X.shape[0]  # 样本数

        # ===================== 梯度计算 =====================
        # 🔥 核心：和DNN一样！！！预测 - 真实
        dZ = y_pred - Y  

        # 权重梯度（和DNN的 dW1, dW2 公式完全一样）
        dW = (1 / m) * np.dot(X.T, dZ)
        # 偏置梯度（和DNN的 db1, db2 公式完全一样）
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

        # ===================== 参数更新（完全一样） =====================
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def train_step(self, X, Y):
        """
        单步训练：前向 + 反向（和DNN完全一样）
        """
        Y = Y.reshape(-1, 1)    # 统一形状
        y_pred = self.forward(X)# 前向
        self.backward(X, Y, y_pred)# 反向更新
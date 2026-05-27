import numpy as np

class TwoLayerDNN:
    def __init__(self, input_dim, hidden_dim, output_dim=1, learning_rate=0.01):
        """
        初始化两层神经网络
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层神经元数量
        :param output_dim: 输出层维度，默认1（二分类）
        :param learning_rate: 学习率
        """
        # 学习率
        self.lr = learning_rate

        # 第一层参数：输入层 -> 隐藏层
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # 权重初始化
        self.b1 = np.zeros((1, hidden_dim))                      # 偏置初始化

        # 第二层参数：隐藏层 -> 输出层
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    @staticmethod
    def _sigmoid(Z):
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _relu(Z):
        """relu激活函数"""
        return np.maximum(0, Z)

    @staticmethod
    def _relu_backward(dA, Z):
        """relu反向传播导数计算"""
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据 (样本数, 输入维度)
        :return: 输出A2 + 中间缓存cache
        """
        # 隐藏层：线性变换 + ReLU激活
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)

        # 输出层：线性变换 + Sigmoid激活（二分类）
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)

        # 缓存前向计算结果，用于反向传播
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def backward(self, X, Y, cache):
        """
        反向传播：计算梯度 + 更新参数
        :param X: 输入数据
        :param Y: 真实标签
        :param cache: 前向传播缓存
        """
        m = X.shape[0]  # 样本数量

        # 从缓存中取值
        Z1, A1, A2 = cache["Z1"], cache["A1"], cache["A2"]

        # ===================== 输出层梯度 =====================
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # ===================== 隐藏层梯度 =====================
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = self._relu_backward(dA1, Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # ===================== 参数更新 =====================
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_step(self, X, Y):
        """单步训练：前向 + 反向 + 参数更新"""
        # 统一标签形状为 (m, 1)
        Y = Y.reshape(-1, 1)
        # 前向传播
        A2, cache = self.forward(X)
        # 反向传播与更新
        self.backward(X, Y, cache)
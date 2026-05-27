import numpy as np

class SimpleAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# 示例：使用 SimpleAdam 优化一个简单的线性回归模型

# 1. 创建模型参数
np.random.seed(42)
w = np.random.randn(3, 1)  # 权重参数
b = np.random.randn(1)     # 偏置参数

# 2. 初始化优化器，传入需要优化的参数列表
optimizer = SimpleAdam(params=[w, b], lr=0.01)

# 3. 生成一些示例数据
X = np.random.randn(100, 3)
y = X.dot(w) + b + 0.1 * np.random.randn(100, 1)  # 添加一些噪声

# 4. 训练循环
for epoch in range(100):
    # 前向传播
    y_pred = X.dot(w) + b
    
    # 计算损失（均方误差）
    loss = np.mean((y_pred - y) ** 2)
    
    # 计算梯度
    grad_w = 2 * X.T.dot(y_pred - y) / len(X)  # w 的梯度
    grad_b = 2 * np.mean(y_pred - y)           # b 的梯度
    
    # 使用优化器更新参数
    optimizer.step(grads=[grad_w, grad_b])
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print(f"Final parameters:")
print(f"w: {w.flatten()}")
print(f"b: {b[0]}")

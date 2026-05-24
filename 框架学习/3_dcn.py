"""
DCN - Deep & Cross Network (Google, 2017)
==========================================
核心思想：用 Cross Network 显式地学习高阶特征交叉
- Cross Network：每一层都在做 x0 * xl 的交叉，层数=k 就是k阶交叉
- Deep Network：和之前一样，隐式泛化

改进了谁？DeepFM
    DeepFM 的问题：FM 只做二阶交叉，高阶交叉全靠 Deep 的黑箱
    DCN 的改进：Cross Network 显式地、有结构地做任意阶交叉

优点：
1. 显式高阶交叉：Cross 层数=k 就是 k 阶交叉，可解释
2. 参数效率高：Cross Network 每层只多 d 个参数（d是特征维度）
3. 不需要人工设计交叉阶数

缺点：
1. Cross Network 的表达能力有限，是 bit-wise 交叉（每个 bit 独立交叉）
2. 所有特征共享同一个交叉权重，不够灵活
"""

import torch
import torch.nn as nn

# ============================================================
# Cross Network 的核心层
# ============================================================
class CrossLayer(nn.Module):
    """
    核心公式：x_{l+1} = x_0 * (w^T * x_l) + b + x_l
              即：用 x0 乘以 xl 的线性变换，实现显式交叉
    """
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        """
        x0: 原始输入（始终不变）
        xl: 上一层 Cross Layer 的输出
        """
        # w^T * x_l -> (batch, 1), 再乘 x0 -> (batch, dim)
        cross = x0 * self.weight(xl) + self.bias + xl
        return cross


# ============================================================
# 完整 DCN 模型
# ============================================================
class DCN(nn.Module):
    def __init__(self, num_features, embed_dim=8, num_cross_layers=3, hidden_dims=[64, 32]):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        input_dim = num_features * embed_dim

        # 共享 Embedding
        self.embedding = nn.Embedding(50, embed_dim)

        # ---- Cross Network：显式高阶交叉 ----
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])

        # ---- Deep Network：隐式泛化 ----
        deep_layers = []
        dim = input_dim
        for h in hidden_dims:
            deep_layers.append(nn.Linear(dim, h))
            deep_layers.append(nn.ReLU())
            dim = h
        self.deep = nn.Sequential(*deep_layers)

        # 最终融合
        self.output_layer = nn.Linear(input_dim + hidden_dims[-1], 1)

    def forward(self, x):
        emb = self.embedding(x)
        x0 = emb.view(emb.size(0), -1)  # (batch, num_feat * embed_dim)

        # Cross Network：逐层交叉，层数=k 即 k 阶
        xl = x0
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        # Deep Network
        deep_out = self.deep(x0)

        # 拼接 + 输出
        combined = torch.cat([xl, deep_out], dim=1)
        return torch.sigmoid(self.output_layer(combined))


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_features = 10
    batch_size = 32

    model = DCN(num_features=num_features, embed_dim=8, num_cross_layers=3, hidden_dims=[64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    x = torch.randint(0, 50, (batch_size, num_features))
    y = torch.randint(0, 2, (batch_size, 1)).float()

    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("模型结构:\n", model)

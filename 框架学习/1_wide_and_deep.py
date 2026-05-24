"""
Wide&Deep (Google, 2016)
========================
核心思想：把"记忆"和"泛化"结合起来
- Wide部分：线性模型，擅长记忆（比如"买过A的人也买了B"这种共现模式）
- Deep部分：DNN，擅长泛化（学到没见过的特征组合）

改进了谁？这是开山之作，没有前序模型。
被谁改进？DeepFM 发现 Wide 部分需要手动设计交叉特征，太麻烦了。

优点：
1. 简单有效，工业界验证过的经典架构
2. 同时具备记忆能力和泛化能力

缺点：
1. Wide 部分的交叉特征需要人工手动设计（feature engineering）
2. Wide 和 Deep 是分开的，没有真正的特征交叉
"""

import torch
import torch.nn as nn

# ============================================================
# 模型定义
# ============================================================
class WideAndDeep(nn.Module):
    def __init__(self, num_features, embed_dim=8, hidden_dims=[64, 32]):
        """
        num_features: 特征数量（比如用户ID、商品ID、年龄等各有各的编号）
        embed_dim:    每个特征的 embedding 维度
        hidden_dims:  Deep 部分的隐藏层维度
        """
        super().__init__()

        # ---- Wide 部分：就是一个线性层 y = wx + b ----
        self.wide = nn.Linear(num_features, 1)

        # ---- Deep 部分：Embedding + MLP ----
        # num_embeddings=50 表示每个特征最多50个不同取值
        self.embedding = nn.Embedding(50, embed_dim)
        deep_input_dim = num_features * embed_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(deep_input_dim, h_dim))
            layers.append(nn.ReLU())
            deep_input_dim = h_dim
        layers.append(nn.Linear(deep_input_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, num_features) 特征索引，比如 [用户ID=1, 商品ID=5, 年龄段=2]
        返回: (batch_size, 1) 预测点击概率的 logit
        """
        # Wide: 直接线性变换（记住历史共现模式）
        wide_out = self.wide(x.float())

        # Deep: 先取 embedding，再过 MLP（泛化到没见过的组合）
        emb = self.embedding(x)                    # (batch, num_feat, embed_dim)
        emb_flat = emb.view(emb.size(0), -1)       # (batch, num_feat * embed_dim)
        deep_out = self.deep(emb_flat)

        # 两部分相加，sigmoid 得到点击概率
        return torch.sigmoid(wide_out + deep_out)


# ============================================================
# 训练演示（用合成数据）
# ============================================================
if __name__ == "__main__":
    # 模拟数据：10个特征，batch_size=32
    num_features = 10
    batch_size = 32

    model = WideAndDeep(num_features=num_features, embed_dim=8, hidden_dims=[64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # 随机生成训练数据
    # 每个特征的取值范围是 0~49（50个不同值），共10个特征
    x = torch.randint(0, 50, (batch_size, num_features))   # 特征索引
    y = torch.randint(0, 2, (batch_size, 1)).float()         # 0/1 标签

    # 训练一步
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("模型结构:\n", model)

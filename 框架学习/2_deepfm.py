"""
DeepFM (Huawei, 2017)
=====================
核心思想：用 FM（因子分解机）替换 Wide&Deep 的 Wide 部分
- FM 部分：自动学习二阶特征交叉，不需要人工设计！
- Deep 部分：和 Wide&Deep 一样，学高阶特征交叉

改进了谁？Wide&Deep
    Wide&Deep 的问题：Wide 部分要手动设计交叉特征（比如"用户性别=男 AND 商品类别=电子"）
    DeepFM 的改进：用 FM 自动学所有二阶交叉，省去人工 feature engineering

优点：
1. 不需要手动设计交叉特征，FM 自动学
2. FM 和 Deep 共享 embedding，参数效率高
3. 同时学到低阶（FM）和高阶（Deep）交叉

缺点：
1. FM 只能做二阶交叉（两两组合），更高阶的交叉靠 Deep 凑
2. Deep 部分的高阶交叉是隐式的，不可解释
"""

import torch
import torch.nn as nn

# ============================================================
# 模型定义
# ============================================================
class DeepFM(nn.Module):
    def __init__(self, num_features, embed_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # ---- FM 部分 ----
        # 一阶权重：每个特征的重要性
        self.fm_first_order = nn.Embedding(50, 1)
        # 二阶 embedding：每个特征一个向量，用来算交叉
        # num_embeddings=50 表示每个特征最多50个不同取值
        self.fm_embedding = nn.Embedding(50, embed_dim)

        # ---- Deep 部分 ----
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
        x: (batch_size, num_features) 特征索引
        """
        # ---- FM 一阶：每个特征自己的贡献 ----
        first_order = self.fm_first_order(x).sum(dim=1)  # (batch, 1)

        # ---- FM 二阶：特征两两交叉 ----
        # 关键公式：Σ Σ <vi, vj> * xi * xj
        # 等价于：(Σ vi)^2 - Σ (vi^2) ，这是计算技巧，O(n)复杂度
        emb = self.fm_embedding(x)                       # (batch, num_feat, embed_dim)
        sum_of_emb = emb.sum(dim=1)                       # (batch, embed_dim)
        sum_of_square = (emb ** 2).sum(dim=1)             # (batch, embed_dim)
        # 交叉项 = 0.5 * [(sum)^2 - sum_of_square]
        cross = 0.5 * (sum_of_emb ** 2 - sum_of_square)  # (batch, embed_dim)
        second_order = cross.sum(dim=1, keepdim=True)     # (batch, 1)

        # ---- Deep 部分：和 Wide&Deep 一样 ----
        emb_flat = emb.view(emb.size(0), -1)
        deep_out = self.deep(emb_flat)

        # 最终输出 = FM一阶 + FM二阶 + Deep
        return torch.sigmoid(first_order + second_order + deep_out)


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_features = 10
    batch_size = 32

    model = DeepFM(num_features=num_features, embed_dim=8, hidden_dims=[64, 32])
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

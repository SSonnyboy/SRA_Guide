"""
SENet 在推荐中的应用 (FiBiNET)
===============================
核心思想：不是所有特征都同等重要，先给特征加权，再做交叉
三步：Squeeze（压缩）→ Excitation（学权重）→ Re-weight（加权）

改进了谁？DeepFM/DCN/AutoInt
    这些模型把所有特征一视同仁地做交叉
    SENet：先让模型判断哪些特征重要，重要的放大，不重要的缩小

优点：
1. 自动学特征重要性，不需要人工判断
2. 即插即用，可以加在任何交叉模型前面
3. 计算开销小（就是两层FC）

缺点：
1. 只学了特征级别的权重，没有改变交叉方式
2. 需要配合其他交叉模块（FM/Cross/Attention）使用
"""

import torch
import torch.nn as nn

# ============================================================
# SENet 核心模块
# ============================================================
class SENet(nn.Module):
    def __init__(self, num_fields, reduction_ratio=4):
        """
        num_fields:       特征域的数量（比如10个特征就是10）
        reduction_ratio:  压缩比，Squeeze后的维度 = num_fields // ratio
        """
        super().__init__()
        reduced_dim = max(num_fields // reduction_ratio, 1)

        # Excitation：两层全连接，学每个特征的重要性
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_dim),   # 先降维
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields),   # 再升回来
            nn.Sigmoid()                          # 输出 0~1 的权重
        )

    def forward(self, x):
        """
        x: (batch, num_fields, embed_dim)
        返回: (batch, num_fields, embed_dim) 加权后的embedding
        """
        # Step 1: Squeeze — 每个特征的embedding压缩成一个数（均值池化）
        # (batch, num_fields, embed_dim) → (batch, num_fields)
        squeeze = x.mean(dim=-1)

        # Step 2: Excitation — 学每个特征的权重
        # (batch, num_fields) → (batch, num_fields)，值在0~1之间
        weights = self.excitation(squeeze)

        # Step 3: Re-weight — 权重乘回原始embedding
        # (batch, num_fields, 1) × (batch, num_fields, embed_dim)
        return x * weights.unsqueeze(-1), weights


# ============================================================
# SENet + Bilinear Interaction（FiBiNET 的完整交叉模块）
# ============================================================
class BilinearInteraction(nn.Module):
    """
    双线性交叉：每对特征之间用一个独立的变换矩阵做交叉
    比FM的内积更灵活（FM共享一个embedding，这里每对特征有独立的W）
    """
    def __init__(self, num_fields, embed_dim):
        super().__init__()
        # 每对特征一个独立的交叉矩阵
        self.bilinear_weights = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim))
            for _ in range(num_fields * (num_fields - 1) // 2)
        ])

    def forward(self, x):
        """
        x: (batch, num_fields, embed_dim)
        返回: 所有特征对的交叉结果拼接
        """
        interactions = []
        idx = 0
        for i in range(x.size(1)):
            for j in range(i + 1, x.size(1)):
                # vi^T × W_ij × vj
                vi = x[:, i, :]                          # (batch, embed_dim)
                vj = x[:, j, :]                          # (batch, embed_dim)
                w = self.bilinear_weights[idx]            # (embed_dim, embed_dim)
                interaction = (vi @ w) * vj               # (batch, embed_dim)
                interactions.append(interaction)
                idx += 1
        return torch.cat(interactions, dim=-1)            # (batch, num_pairs * embed_dim)


# ============================================================
# 完整模型：SENet + Bilinear + Deep
# ============================================================
class FiBiNET(nn.Module):
    def __init__(self, num_fields, embed_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.num_fields = num_fields

        # Embedding
        self.embedding = nn.Embedding(50, embed_dim)

        # SENet：特征加权
        self.senet = SENet(num_fields)

        # Bilinear交叉：对加权后的特征做双线性交叉
        num_pairs = num_fields * (num_fields - 1) // 2
        self.bilinear = BilinearInteraction(num_fields, embed_dim)

        # Deep部分
        deep_input_dim = num_fields * embed_dim
        layers = []
        dim = deep_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        self.deep = nn.Sequential(*layers)

        # 输出层：Bilinear交叉 + Deep
        self.output_layer = nn.Linear(num_pairs * embed_dim + hidden_dims[-1], 1)

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)  # (batch, num_fields, embed_dim)

        # SENet加权
        weighted_emb, weights = self.senet(emb)  # (batch, num_fields, embed_dim)

        # Bilinear交叉
        bi_out = self.bilinear(weighted_emb)     # (batch, num_pairs * embed_dim)

        # Deep
        emb_flat = emb.view(emb.size(0), -1)
        deep_out = self.deep(emb_flat)

        # 融合
        combined = torch.cat([bi_out, deep_out], dim=1)
        return torch.sigmoid(self.output_layer(combined))


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_fields = 10   # 10个特征域
    batch_size = 32
    embed_dim = 8

    model = FiBiNET(num_fields=num_fields, embed_dim=embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    x = torch.randint(0, 50, (batch_size, num_fields))
    y = torch.randint(0, 2, (batch_size, 1)).float()

    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    # 看看 SENet 学到的特征权重
    with torch.no_grad():
        emb = model.embedding(x)
        _, weights = model.senet(emb)
        print(f"Loss: {loss.item():.4f}")
        print(f"SENet学到的特征权重（第1个样本）: {weights[0].numpy().round(3)}")
        print(f"权重越大 → 该特征越重要")
        print(f"\n模型结构:\n{model}")

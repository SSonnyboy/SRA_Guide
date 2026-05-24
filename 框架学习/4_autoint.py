"""
AutoInt (2019)
==============
核心思想：用 Multi-Head Self-Attention 自动学习特征交叉
- 把每个特征的 embedding 当作一个 token
- 用 Transformer 的 Self-Attention 让特征之间互相"看"
- 多头注意力 = 同时从多个角度学交叉关系

改进了谁？DCN
    DCN 的问题：Cross Network 是 bit-wise 交叉，每个 bit 独立，不够灵活
    AutoInt 的改进：Self-Attention 让整个 embedding 向量参与交叉，而且不同特征对之间的交叉权重是动态计算的（不是共享的）

优点：
1. 动态交叉权重：注意力权重根据具体样本计算，不同样本有不同的交叉方式
2. 多头注意力：从多个角度学交叉（比如一个头学"性别-品类"，另一个头学"年龄-价格"）
3. 可解释性：注意力权重能看出哪些特征对最重要
4. 端到端学习，不需要人工设计

缺点：
1. 计算量大：Self-Attention 是 O(n^2)，特征多时开销大
2. 实际效果提升不一定明显，在某些数据集上不如 DCN
3. 过拟合风险更高（参数更多）
"""

import torch
import torch.nn as nn
import math

# ============================================================
# Multi-Head Self-Attention 特征交叉层
# ============================================================
class MultiHeadAttentionCross(nn.Module):
    """
    把每个特征的 embedding 当作一个 token：
      Query = 特征i的embedding
      Key   = 特征j的embedding
      Value = 特征j的embedding
    注意力权重 = Q·K / sqrt(d)，表示特征i和j的交叉重要性
    """
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, num_features, embed_dim)
        返回: (batch, num_features, embed_dim) 交叉后的特征表示
        """
        batch_size, num_feat, embed_dim = x.shape

        # 线性变换 -> 多头拆分
        Q = self.W_Q(x).view(batch_size, num_feat, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, num_feat, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, num_feat, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (batch, heads, num_feat, head_dim)

        # 注意力分数：特征之间的相似度/重要性
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)  # (batch, heads, num_feat, num_feat)

        # 加权聚合：每个特征吸收其他特征的信息（= 交叉）
        out = torch.matmul(attn, V)  # (batch, heads, num_feat, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_feat, embed_dim)
        return self.W_O(out)


# ============================================================
# 完整 AutoInt 模型
# ============================================================
class AutoInt(nn.Module):
    def __init__(self, num_features, embed_dim=8, num_heads=2, num_attn_layers=3, hidden_dims=[64, 32]):
        super().__init__()
        self.embedding = nn.Embedding(50, embed_dim)

        # ---- Attention 交叉部分 ----
        self.attn_layers = nn.ModuleList([
            MultiHeadAttentionCross(embed_dim, num_heads) for _ in range(num_attn_layers)
        ])
        self.attn_relu = nn.ReLU()

        # ---- Deep 部分（可选，和 DCN 一样加个 DNN）----
        deep_input_dim = num_features * embed_dim
        layers = []
        dim = deep_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        self.deep = nn.Sequential(*layers)

        # 最终输出：Attention交叉结果 + Deep结果
        self.output_layer = nn.Linear(num_features * embed_dim + hidden_dims[-1], 1)

    def forward(self, x):
        emb = self.embedding(x)  # (batch, num_feat, embed_dim)

        # 多层 Self-Attention 交叉
        attn_out = emb
        for attn_layer in self.attn_layers:
            attn_out = self.attn_relu(attn_layer(attn_out) + attn_out)  # 残差连接

        attn_flat = attn_out.view(attn_out.size(0), -1)

        # Deep 部分
        emb_flat = emb.view(emb.size(0), -1)
        deep_out = self.deep(emb_flat)

        # 融合
        combined = torch.cat([attn_flat, deep_out], dim=1)
        return torch.sigmoid(self.output_layer(combined))


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_features = 10
    batch_size = 32

    model = AutoInt(num_features=num_features, embed_dim=8, num_heads=2, num_attn_layers=3)
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

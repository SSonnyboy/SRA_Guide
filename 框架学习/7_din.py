"""
DIN - Deep Interest Network (Alibaba, 2018)
============================================
核心思想：用户的历史行为不是同等重要的，要根据候选物品动态加权
- 用户看过：[手机, 衣服, 鞋子, 电脑, 零食]
- 候选物品：键盘
- DIN的做法：用注意力机制算每个历史行为和"键盘"的相似度
  → 电脑(0.8), 手机(0.6), 鞋子(0.1), 衣服(0.1), 零食(0.05)
  → 用户对键盘的兴趣 = 0.8×电脑 + 0.6×手机 + 0.1×鞋子 + ...

改进了谁？传统方法（DeepFM/DCN）
    传统方法：把用户历史行为的embedding直接求和/平均 → 丢失了兴趣多样性
    DIN：根据候选物品动态加权历史行为 → 保留了用户多样的兴趣

优点：
1. 兴趣多样性：不同候选物品激活不同的历史行为
2. 可解释性：注意力权重能看出"用户因为看了什么而点击这个"
3. 工业验证：阿里巴巴大规模部署，效果显著

缺点：
1. 只关注行为的相关性，忽略了行为之间的时间顺序
2. 长序列计算开销大（每个行为都要和候选物品算注意力）
"""

import torch
import torch.nn as nn

# ============================================================
# DIN 注意力模块
# ============================================================
class DINAttention(nn.Module):
    """
    核心：用一个小型DNN计算 历史行为 和 候选物品 的相关性分数
    输入：(行为, 候选物品) 对
    输出：注意力分数（0~1）
    """
    def __init__(self, embed_dim):
        super().__init__()
        # 注意力网络：输入 = [行为, 候选, 行为-候选, 行为*候选]
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, candidate, behavior, behavior_mask=None):
        """
        candidate: (batch, 1, embed_dim)    候选物品的embedding
        behavior:  (batch, seq_len, embed_dim) 用户历史行为序列
        behavior_mask: (batch, seq_len)     哪些位置是padding（True=有效）
        返回: (batch, embed_dim) 加权后的用户兴趣表示
        """
        seq_len = behavior.size(1)

        # 把候选物品扩展到和行为序列一样长
        # (batch, 1, dim) → (batch, seq_len, dim)
        candidate_expanded = candidate.expand(-1, seq_len, -1)

        # 拼接4种特征：原始、差、乘、原始
        # 这是DIN论文的技巧，给注意力网络更多信息
        attention_input = torch.cat([
            behavior,                               # 行为本身
            candidate_expanded,                     # 候选物品
            behavior - candidate_expanded,          # 差异
            behavior * candidate_expanded,          # 交互
        ], dim=-1)  # (batch, seq_len, embed_dim * 4)

        # 计算注意力分数
        scores = self.attention_net(attention_input).squeeze(-1)  # (batch, seq_len)

        # Mask：padding位置设为-inf，softmax后变成0
        if behavior_mask is not None:
            scores = scores.masked_fill(~behavior_mask, float('-inf'))

        # Softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)

        # 加权求和
        # (batch, seq_len, 1) × (batch, seq_len, dim) → sum → (batch, dim)
        user_interest = (attn_weights.unsqueeze(-1) * behavior).sum(dim=1)

        return user_interest, attn_weights


# ============================================================
# 完整 DIN 模型
# ============================================================
class DIN(nn.Module):
    def __init__(self, num_items, embed_dim=8, hidden_dims=[64, 32], max_seq_len=20):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.max_seq_len = max_seq_len

        # 注意力模块
        self.attention = DINAttention(embed_dim)

        # Deep网络：输入 = 用户兴趣 + 候选物品 + 用户画像特征
        deep_input_dim = embed_dim * 3  # user_interest + candidate + context
        layers = []
        dim = deep_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, candidate_id, behavior_ids, behavior_mask):
        """
        candidate_id:   (batch,) 候选物品ID
        behavior_ids:   (batch, seq_len) 用户历史行为序列
        behavior_mask:  (batch, seq_len) 有效位置标记
        """
        # Embedding
        candidate_emb = self.item_embedding(candidate_id).unsqueeze(1)  # (batch, 1, dim)
        behavior_emb = self.item_embedding(behavior_ids)                 # (batch, seq_len, dim)

        # 注意力：根据候选物品加权历史行为
        user_interest, attn_weights = self.attention(candidate_emb, behavior_emb, behavior_mask)

        # 拼接特征
        combined = torch.cat([
            user_interest,
            candidate_emb.squeeze(1),
            user_interest * candidate_emb.squeeze(1)  # 交叉特征
        ], dim=-1)

        # 预测
        return torch.sigmoid(self.deep(combined)), attn_weights


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_items = 100
    batch_size = 4
    seq_len = 10

    model = DIN(num_items=num_items, embed_dim=8, hidden_dims=[64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # 模拟数据
    candidate_id = torch.randint(1, num_items, (batch_size,))
    behavior_ids = torch.randint(1, num_items, (batch_size, seq_len))
    behavior_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # 让某些位置是padding
    behavior_mask[:, 7:] = False
    behavior_ids[:, 7:] = 0

    y = torch.randint(0, 2, (batch_size, 1)).float()

    pred, attn_weights = model(candidate_id, behavior_ids, behavior_mask)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print(f"\n注意力权重（第1个样本，前7个行为）:")
    print(attn_weights[0][:7].detach().numpy().round(3))
    print(f"→ 权重大的行为 = 和候选物品最相关的历史行为")

"""
DIEN - Deep Interest Evolution Network (Alibaba, 2019)
=======================================================
核心思想：用户的兴趣是随时间演变的，不能只看相关性，还要看变化趋势
- DIN的问题：只考虑"历史行为和候选物品相不相关"，忽略了时间顺序
- DIEN的做法：
  1. 用GRU建模行为序列的时序关系 → 兴趣演化过程
  2. 用注意力机制（AUGRU）在GRU的基础上加权 → 关注和候选物品相关的演化路径

改进了谁？DIN
    DIN：把历史行为当成无序集合，只算注意力
    DIEN：用GRU建模行为之间的时序关系，兴趣是"演化"的

优点：
1. 捕捉兴趣演化：知道用户的兴趣是怎么变化的
2. AUGRU：在时序建模的基础上再加注意力，两全其美
3. 更精准：比如用户从"看便宜货"演化到"看高端品"，DIEN能学到这个趋势

缺点：
1. GRU计算开销大，序列长时训练慢
2. GRU的长期依赖能力有限（比不上Transformer）
3. 兴趣漂移（interest drift）很难建模
"""

import torch
import torch.nn as nn

# ============================================================
# AUGRU: 带注意力的GRU
# ============================================================
class AUGRUCell(nn.Module):
    """
    标准GRU：根据当前输入和上一时刻隐藏状态，更新隐藏状态
    AUGRU：在GRU的更新门上乘以注意力权重
      → 如果当前行为和候选物品相关（注意力高），就多更新隐藏状态
      → 如果不相关（注意力低），就少更新，保留之前的兴趣
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # GRU的三个门
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)  # 更新门
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)  # 重置门
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)  # 候选隐藏状态

    def forward(self, x, h_prev, attn_score):
        """
        x:         (batch, input_dim)   当前时刻的行为embedding
        h_prev:    (batch, hidden_dim)  上一时刻的隐藏状态
        attn_score: (batch, 1)         当前行为对候选物品的注意力分数
        返回:      (batch, hidden_dim)  当前时刻的隐藏状态
        """
        combined = torch.cat([x, h_prev], dim=-1)

        # 标准GRU门
        z = torch.sigmoid(self.W_z(combined))   # 更新门 控制采纳多少新信息
        r = torch.sigmoid(self.W_r(combined))   # 重置门 采纳多少旧知识？
        h_tilde = torch.tanh(self.W_h(torch.cat([x, r * h_prev], dim=-1)))  # 新信息

        # AUGRU的关键：更新门 × 注意力分数
        # 注意力高 → 更新门大 → 多采纳新信息
        # 注意力低 → 更新门小 → 保留旧兴趣
        z_attn = z * attn_score  # 注意力加权的更新门

        # 更新隐藏状态
        h = (1 - z_attn) * h_prev + z_attn * h_tilde
        return h


# ============================================================
# 兴趣演化层
# ============================================================
class InterestEvolutionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.augru = AUGRUCell(embed_dim, hidden_dim)
        # 注意力网络：计算每个行为和候选物品的相关性
        self.attn_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, behavior_seq, candidate, mask):
        """
        behavior_seq: (batch, seq_len, embed_dim) 用户历史行为序列
        candidate:    (batch, embed_dim) 候选物品
        mask:         (batch, seq_len) 有效位置标记
        """
        batch_size, seq_len, embed_dim = behavior_seq.shape
        h = torch.zeros(batch_size, self.augru.hidden_dim, device=behavior_seq.device)

        # 存储每个时刻的隐藏状态
        h_seq = []

        for t in range(seq_len):
            # 当前行为
            x_t = behavior_seq[:, t, :]  # (batch, embed_dim)

            # 计算当前行为和候选物品的注意力分数
            attn_input = torch.cat([x_t, candidate], dim=-1)  # (batch, embed_dim*2)
            attn_score = torch.sigmoid(self.attn_fc(attn_input))  # (batch, 1)

            # AUGRU更新
            h = self.augru(x_t, h, attn_score)
            h_seq.append(h)

        # (batch, seq_len, hidden_dim)
        h_seq = torch.stack(h_seq, dim=1)

        # 取最后一个有效时刻的隐藏状态作为用户兴趣
        # 用mask找到每个样本最后一个有效位置
        last_idx = mask.sum(dim=1).long() - 1  # (batch,)
        last_idx = last_idx.clamp(min=0)
        user_interest = h_seq[torch.arange(batch_size), last_idx]  # (batch, hidden_dim)

        return user_interest


# ============================================================
# 完整 DIEN 模型
# ============================================================
class DIEN(nn.Module):
    def __init__(self, num_items, embed_dim=8, hidden_dim=16, mlp_dims=[64, 32]):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # 兴趣演化层
        self.interest_evolution = InterestEvolutionLayer(embed_dim, hidden_dim)

        # 输出MLP
        layers = []
        dim = hidden_dim + embed_dim  # user_interest + candidate
        for h in mlp_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, candidate_id, behavior_ids, mask):
        candidate_emb = self.item_embedding(candidate_id)      # (batch, dim)
        behavior_emb = self.item_embedding(behavior_ids)        # (batch, seq, dim)

        # 兴趣演化
        user_interest = self.interest_evolution(behavior_emb, candidate_emb, mask)

        # 预测
        combined = torch.cat([user_interest, candidate_emb], dim=-1)
        return torch.sigmoid(self.mlp(combined))


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_items = 100
    batch_size = 4
    seq_len = 10

    model = DIEN(num_items=num_items, embed_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    candidate_id = torch.randint(1, num_items, (batch_size,))
    behavior_ids = torch.randint(1, num_items, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, 7:] = False
    behavior_ids[:, 7:] = 0

    y = torch.randint(0, 2, (batch_size, 1)).float()

    pred = model(candidate_id, behavior_ids, mask)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print(f"\n模型结构:\n{model}")

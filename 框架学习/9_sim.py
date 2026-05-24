"""
SIM - Search-based Interest Model (Alibaba, 2020)
===================================================
核心思想：超长行为序列下，先搜索再建模，降低计算量
- DIN/DIEN的问题：序列长到几千上万时，注意力计算量太大（O(n)）
- SIM的做法：
  1. GSU（General Search Unit）：用检索快速找到和候选物品相关的历史行为（从1万缩到100）
  2. ESU（Exact Search Unit）：对筛选后的行为用注意力精确建模

改进了谁？DIN / DIEN
    DIN/DIEN：对所有历史行为都算注意力，序列长时计算量爆炸
    SIM：先粗筛再精算，可以处理超长序列（1万+行为）

优点：
1. 支持超长序列：可以建模用户几个月甚至几年的行为
2. 两阶段效率高：GSU粗筛很快，ESU只处理少量行为
3. 灵活：GSU可以用各种检索方法（分类、向量检索等）

缺点：
1. GSU粗筛可能漏掉重要行为（召回率问题）
2. 两阶段设计增加了系统复杂度
3. 实时性要求高时，检索延迟可能成为瓶颈
"""

import torch
import torch.nn as nn

# ============================================================
# GSU: 粗筛阶段 — 快速找到相关的候选行为
# ============================================================
class SoftSearchGSU(nn.Module):
    """
    软搜索：用一个简单的FC网络计算每个历史行为和候选物品的相似度
    工业界也有硬搜索：直接用类别/标签过滤（更快但更粗糙）
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, candidate_emb, behavior_emb, top_k):
        """
        candidate_emb: (batch, embed_dim)
        behavior_emb:  (batch, long_seq_len, embed_dim)
        top_k: 保留最相关的k个行为
        返回: (batch, top_k, embed_dim) 筛选后的行为
        """
        # 计算每个行为和候选物品的相似度
        # (batch, long_seq, dim) * (batch, dim, 1) → (batch, long_seq, 1)
        scores = (behavior_emb * candidate_emb.unsqueeze(1)).sum(dim=-1)  # (batch, long_seq)

        # 取top-k
        topk_scores, topk_indices = scores.topk(top_k, dim=1)  # (batch, top_k)

        # 收集top-k对应的embedding
        # 需要扩展indices维度来gather
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, behavior_emb.size(-1))
        topk_behavior = torch.gather(behavior_emb, 1, topk_indices_expanded)  # (batch, top_k, dim)

        return topk_behavior, topk_indices


# ============================================================
# ESU: 精算阶段 — 对筛选后的行为用注意力建模
# ============================================================
class ESU(nn.Module):
    """和DIN一样的注意力机制，但输入是GSU筛选后的行为"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, candidate_emb, filtered_behavior):
        """
        candidate_emb:      (batch, embed_dim)
        filtered_behavior:  (batch, top_k, embed_dim) GSU筛选后的行为
        """
        top_k = filtered_behavior.size(1)
        candidate_exp = candidate_emb.unsqueeze(1).expand(-1, top_k, -1)

        # 拼接4种特征
        attn_input = torch.cat([
            filtered_behavior,
            candidate_exp,
            filtered_behavior - candidate_exp,
            filtered_behavior * candidate_exp,
        ], dim=-1)

        # 注意力分数
        scores = self.attn_net(attn_input).squeeze(-1)  # (batch, top_k)
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        user_interest = (attn_weights.unsqueeze(-1) * filtered_behavior).sum(dim=1)

        return user_interest, attn_weights


# ============================================================
# 完整 SIM 模型
# ============================================================
class SIM(nn.Module):
    def __init__(self, num_items, embed_dim=8, top_k=5, mlp_dims=[64, 32]):
        super().__init__()
        self.top_k = top_k
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # 两阶段
        self.gsu = SoftSearchGSU(embed_dim)
        self.esu = ESU(embed_dim)

        # 输出MLP
        layers = []
        dim = embed_dim * 2  # user_interest + candidate
        for h in mlp_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, candidate_id, long_behavior_ids, mask=None):
        """
        candidate_id:       (batch,) 候选物品ID
        long_behavior_ids:  (batch, long_seq_len) 超长行为序列
        """
        candidate_emb = self.item_embedding(candidate_id)         # (batch, dim)
        behavior_emb = self.item_embedding(long_behavior_ids)      # (batch, long_seq, dim)

        # ---- Stage 1: GSU 粗筛 ----
        filtered_behavior, topk_indices = self.gsu(candidate_emb, behavior_emb, self.top_k)

        # ---- Stage 2: ESU 精算 ----
        user_interest, attn_weights = self.esu(candidate_emb, filtered_behavior)

        # 预测
        combined = torch.cat([user_interest, candidate_emb], dim=-1)
        return torch.sigmoid(self.mlp(combined)), attn_weights, topk_indices


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    num_items = 200
    batch_size = 4
    long_seq_len = 50  # 模拟长序列（实际可能是几千）
    top_k = 8          # 粗筛保留8个

    model = SIM(num_items=num_items, embed_dim=8, top_k=top_k, mlp_dims=[64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    candidate_id = torch.randint(1, num_items, (batch_size,))
    long_behavior_ids = torch.randint(1, num_items, (batch_size, long_seq_len))
    y = torch.randint(0, 2, (batch_size, 1)).float()

    pred, attn_weights, topk_indices = model(candidate_id, long_behavior_ids)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print(f"\nGSU筛选的top-{top_k}行为位置（第1个样本）:")
    print(f"  从{long_seq_len}个行为中选出: {topk_indices[0].tolist()}")
    print(f"\nESU注意力权重（第1个样本）:")
    print(f"  {attn_weights[0].detach().numpy().round(3)}")
    print(f"\n→ 两阶段: 先粗筛到{top_k}个，再精算注意力")

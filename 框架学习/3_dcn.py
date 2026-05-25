"""
DCN & DCN-V2
=============

【DCN (Google, 2017)】
核心思想：用 Cross Network 显式地学习高阶特征交叉
- 公式：x_{l+1} = x_0 × (w^T × x_l) + b + x_l
- w^T × x_l 是标量 → bit-wise 交叉，所有维度共享同一个权重

优点：
1. 显式高阶交叉：Cross 层数=k 就是 k 阶交叉
2. 参数效率高：每层只多 d 个参数

缺点：
1. bit-wise 交叉，表达能力有限
2. 所有特征维度共享同一个交叉权重

【DCN-V2 (Google, 2020)】
改进了谁？DCN
    DCN的问题：w^T × x_l 是标量 → bit-wise，太粗糙
    DCN-V2的改进：用矩阵 W 替代向量 w → vector-wise 交叉

两种改进方案：
  方案1：W 是全秩矩阵（参数多，表达强）
  方案2：W 是低秩矩阵分解（参数少，效率高）

优点：
1. vector-wise 交叉，表达能力更强
2. 低秩版本参数可控
3. 可以用 MOE 进一步增强

缺点：
1. 全秩版本参数量大
2. 实际效果提升不一定显著
"""

import torch
import torch.nn as nn


# ============================================================
# DCN (2017): Cross Layer — bit-wise 交叉
# ============================================================
class CrossLayer(nn.Module):
    """
    公式：x_{l+1} = x_0 × (w^T × x_l) + b + x_l
    w^T × x_l 是标量 → 所有维度乘同一个数 → bit-wise
    """
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        cross = x0 * self.weight(xl) + self.bias + xl
        return cross


# ============================================================
# DCN-V2 (2020): Cross Layer — vector-wise 交叉
# ============================================================
class CrossLayerV2(nn.Module):
    """
    公式：x_{l+1} = x_0 × (W × x_l) + b + x_l
    W × x_l 是向量（不是标量）→ 每个维度有不同的交叉权重 → vector-wise

    两种W的选择：
      - 全秩：W 是 (d, d) 矩阵，参数多，表达强
      - 低秩：W = U × V^T，U是(d, r), V是(d, r)，参数少
    """
    def __init__(self, input_dim, low_rank=None):
        super().__init__()
        if low_rank is None:
            # 全秩版本：W 是完整的 d×d 矩阵
            self.W = nn.Linear(input_dim, input_dim, bias=False)
        else:
            # 低秩版本：W = U × V^T，参数从 d² 降到 2×d×r
            # V: input_dim → low_rank（先降维）
            # U: low_rank → input_dim（再升维）
            self.V = nn.Linear(input_dim, low_rank, bias=False)
            self.U = nn.Linear(low_rank, input_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.use_low_rank = low_rank is not None

    def forward(self, x0, xl):
        if self.use_low_rank:
            # 低秩：W × x_l = U × (V^T × x_l)
            Wxl = self.U(self.V(xl))  # (batch, dim) → 向量
        else:
            # 全秩：W × x_l
            Wxl = self.W(xl)           # (batch, dim) → 向量

        # x_0 × (W × x_l)：向量逐元素相乘，不是标量！
        cross = x0 * Wxl + self.bias + xl
        return cross


# ============================================================
# DCN-V2 + MOE 版本（论文中的增强方案）
# ============================================================
class CrossLayerV2MOE(nn.Module):
    """
    用多个CrossLayer作为专家，门控网络选择专家
    进一步增强表达能力
    """
    def __init__(self, input_dim, num_experts=4, low_rank=None):
        super().__init__()
        self.experts = nn.ModuleList([
            CrossLayerV2(input_dim, low_rank) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x0, xl):
        gate_weights = self.gate(x0)  # (batch, num_experts)
        expert_outputs = torch.stack([exp(x0, xl) for exp in self.experts], dim=1)
        # (batch, num_experts, dim)
        # 加权组合
        output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return output


# ============================================================
# 完整 DCN 模型（同时支持 v1 和 v2）
# ============================================================
class DCN(nn.Module):
    def __init__(self, num_features, embed_dim=8, num_cross_layers=3,
                 hidden_dims=[64, 32], version='v1', low_rank=None, num_experts=None):
        """
        version: 'v1' 或 'v2'
        low_rank: 低秩维度（None=全秩）
        num_experts: MOE专家数（None=不用MOE）
        """
        super().__init__()
        self.version = version
        input_dim = num_features * embed_dim

        # 共享 Embedding
        self.embedding = nn.Embedding(50, embed_dim)

        # ---- Cross Network ----
        self.cross_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            if version == 'v1':
                self.cross_layers.append(CrossLayer(input_dim))
            elif num_experts is not None:
                self.cross_layers.append(CrossLayerV2MOE(input_dim, num_experts, low_rank))
            else:
                self.cross_layers.append(CrossLayerV2(input_dim, low_rank))

        # ---- Deep Network ----
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
        x0 = emb.view(emb.size(0), -1)

        xl = x0
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        deep_out = self.deep(x0)
        combined = torch.cat([xl, deep_out], dim=1)
        return torch.sigmoid(self.output_layer(combined))


# ============================================================
# 训练演示：对比三个版本
# ============================================================
if __name__ == "__main__":
    num_features = 10
    batch_size = 32
    x = torch.randint(0, 50, (batch_size, num_features))
    y = torch.randint(0, 2, (batch_size, 1)).float()
    loss_fn = nn.BCELoss()

    # ---- DCN v1 ----
    model_v1 = DCN(num_features, version='v1')
    optimizer = torch.optim.Adam(model_v1.parameters(), lr=0.001)
    pred = model_v1(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"DCN-v1 Loss: {loss.item():.4f}")

    # ---- DCN v2（全秩）----
    model_v2 = DCN(num_features, version='v2')
    optimizer = torch.optim.Adam(model_v2.parameters(), lr=0.001)
    pred = model_v2(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"DCN-v2（全秩）Loss: {loss.item():.4f}")

    # ---- DCN v2（低秩）----
    model_v2_lr = DCN(num_features, version='v2', low_rank=4)
    optimizer = torch.optim.Adam(model_v2_lr.parameters(), lr=0.001)
    pred = model_v2_lr(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"DCN-v2（低秩r=4）Loss: {loss.item():.4f}")

    # ---- DCN v2 + MOE ----
    model_v2_moe = DCN(num_features, version='v2', num_experts=4)
    optimizer = torch.optim.Adam(model_v2_moe.parameters(), lr=0.001)
    pred = model_v2_moe(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"DCN-v2+MOE Loss: {loss.item():.4f}")

    # ---- 参数量对比 ----
    print(f"\n参数量对比:")
    print(f"  DCN-v1:       {sum(p.numel() for p in model_v1.parameters()):>8,}")
    print(f"  DCN-v2 全秩:  {sum(p.numel() for p in model_v2.parameters()):>8,}")
    print(f"  DCN-v2 低秩:  {sum(p.numel() for p in model_v2_lr.parameters()):>8,}")
    print(f"  DCN-v2+MOE:   {sum(p.numel() for p in model_v2_moe.parameters()):>8,}")

"""
MMOE - Multi-gate Mixture of Experts (Google, 2018)
=====================================================
核心思想：每个任务一个独立的门控网络，让不同任务选择不同的专家组合
- MOE的问题：所有任务共享同一个门控，任务需求冲突时无法调和
- MMOE的改进：每个任务有自己的门控，独立选择专家

改进了谁？MOE
    MOE：所有任务共享同一个门控 → 任务冲突时无法兼顾
    MMOE：每个任务独立门控 → 任务A可能选专家1+2，任务B选专家3+4

优点：
1. 每个任务独立选择专家，彻底解决任务冲突
2. 门控网络参数少，计算开销小
3. 工业界广泛验证（Google、YouTube等）

缺点：
1. 专家数量需要调参
2. 如果任务太多，门控可能退化（所有门控学到一样的权重）
3. 专家之间没有显式的多样性约束
"""

import torch
import torch.nn as nn

# ============================================================
# 专家网络
# ============================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = dim

    def forward(self, x):
        return self.net(x)


# ============================================================
# MMOE 模型
# ============================================================
class MMOE(nn.Module):
    def __init__(self, input_dim, expert_dims, num_experts, num_tasks):
        """
        input_dim:   输入维度
        expert_dims: 专家网络维度 [128, 64]
        num_experts: 专家数量
        num_tasks:   任务数量
        """
        super().__init__()

        # 共享专家池
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dims) for _ in range(num_experts)
        ])

        # 每个任务一个独立的门控（和MOE的区别！MOE只有一个门控）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

        # 每个任务的输出层
        expert_output_dim = expert_dims[-1]
        self.task_heads = nn.ModuleList([
            nn.Linear(expert_output_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        """
        x: (batch, input_dim)
        返回: list of (batch, 1), list of gate weights
        """
        # 所有专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        # (num_experts, batch, expert_output_dim)

        outputs = []
        all_gate_weights = []

        for i in range(len(self.gates)):
            # 第i个任务的门控权重
            gate_weights = self.gates[i](x)  # (batch, num_experts)
            all_gate_weights.append(gate_weights)

            # 加权组合专家输出
            # (batch, num_experts) × (num_experts, batch, dim) → (batch, dim)
            weighted = torch.einsum('be,ebd->bd', gate_weights, expert_outputs)

            # 任务预测
            outputs.append(torch.sigmoid(self.task_heads[i](weighted)))

        return outputs, all_gate_weights


# ============================================================
# 对比：MOE vs MMOE
# ============================================================
class MOE(nn.Module):
    """MOE：所有任务共享一个门控"""
    def __init__(self, input_dim, expert_dims, num_experts, num_tasks):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dims) for _ in range(num_experts)
        ])
        # 只有一个门控！所有任务共享
        self.shared_gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        expert_output_dim = expert_dims[-1]
        self.task_heads = nn.ModuleList([
            nn.Linear(expert_output_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=0)
        gate_weights = self.shared_gate(x)  # 所有任务用同一个门控

        outputs = []
        for head in self.task_heads:
            weighted = torch.einsum('be,ebd->bd', gate_weights, expert_outputs)
            outputs.append(torch.sigmoid(head(weighted)))
        return outputs, gate_weights


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    input_dim = 32
    num_experts = 4
    num_tasks = 3
    batch_size = 64

    x = torch.randn(batch_size, input_dim)
    labels = [torch.randint(0, 2, (batch_size, 1)).float() for _ in range(num_tasks)]

    # ---- MMOE ----
    mmoe = MMOE(input_dim, [128, 64], num_experts, num_tasks)
    optimizer = torch.optim.Adam(mmoe.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    preds, gates = mmoe(x)
    loss = sum(loss_fn(preds[i], labels[i]) for i in range(num_tasks))
    loss.backward()
    optimizer.step()

    print("=== MMOE（每个任务独立门控）===")
    print(f"总Loss: {loss.item():.4f}")
    print(f"\n任务0的门控: {gates[0][0].detach().numpy().round(3)}")
    print(f"任务1的门控: {gates[1][0].detach().numpy().round(3)}")
    print(f"任务2的门控: {gates[2][0].detach().numpy().round(3)}")
    print(f"→ 每个任务选择不同的专家组合！")

    # ---- MOE ----
    print("\n=== MOE（所有任务共享门控）===")
    moe = MOE(input_dim, [128, 64], num_experts, num_tasks)
    preds, gate = moe(x)
    print(f"共享门控: {gate[0].detach().numpy().round(3)}")
    print(f"→ 所有任务被迫用同一套权重，任务冲突时无法调和")

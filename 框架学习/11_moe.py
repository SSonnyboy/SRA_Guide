"""
MOE - Mixture of Experts (多专家混合)
======================================
核心思想：用多个"专家网络"代替单一共享底层，门控网络决定用哪些专家
- 专家网络：每个专家擅长提取不同类型的特征
- 门控网络：根据输入决定每个专家的权重

改进了谁？Shared-Bottom
    Shared-Bottom的问题：所有任务被迫用同一个底层，负迁移
    MOE的改进：多个专家提供不同表示，门控选择合适的专家组合

优点：
1. 多专家提供多样性，减少负迁移
2. 门控网络自动选择专家，灵活
3. 比Shared-Bottom表达能力更强

缺点：
1. 所有任务共享同一个门控 → 任务需求冲突时无法调和
2. 这正是MMOE要解决的问题（每个任务独立门控）
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
# 门控网络
# ============================================================
class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # 输出每个专家的权重（softmax归一化）
        return torch.softmax(self.fc(x), dim=-1)  # (batch, num_experts)


# ============================================================
# MOE 模型
# ============================================================
class MOE(nn.Module):
    def __init__(self, input_dim, expert_dims, num_experts, num_tasks):
        """
        input_dim:   输入维度
        expert_dims: 每个专家的隐藏层维度 [128, 64]
        num_experts: 专家数量（比如4个）
        num_tasks:   任务数量
        """
        super().__init__()

        # 多个专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dims) for _ in range(num_experts)
        ])

        # 只有一个门控！所有任务共享（这是MOE和MMOE的核心区别）
        self.gate = Gate(input_dim, num_experts)

        # 每个任务的输出层
        expert_output_dim = expert_dims[-1]
        self.task_heads = nn.ModuleList([
            nn.Linear(expert_output_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        """
        x: (batch, input_dim)
        返回: list of (batch, 1)
        """
        # 所有专家的输出
        expert_outputs = [expert(x) for expert in self.experts]
        # (num_experts, batch, expert_output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # 共享门控：所有任务用同一套权重
        gate_weights = self.gate(x)  # (batch, num_experts)

        # 加权组合专家输出
        # (batch, num_experts) × (num_experts, batch, dim) → (batch, dim)
        weighted_output = torch.einsum(
            'be,ebd->bd',
            gate_weights,
            expert_outputs
        )

        # 每个任务头预测（但共享同一个加权结果）
        outputs = []
        for head in self.task_heads:
            outputs.append(torch.sigmoid(head(weighted_output)))

        return outputs, gate_weights


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    input_dim = 32
    num_experts = 4
    num_tasks = 3

    model = MOE(
        input_dim=input_dim,
        expert_dims=[128, 64],
        num_experts=num_experts,
        num_tasks=num_tasks
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    batch_size = 64
    x = torch.randn(batch_size, input_dim)
    labels = [torch.randint(0, 2, (batch_size, 1)).float() for _ in range(num_tasks)]

    preds, gate_weights = model(x)

    total_loss = 0
    for i in range(num_tasks):
        total_loss += loss_fn(preds[i], labels[i])

    total_loss.backward()
    optimizer.step()

    print(f"总Loss: {total_loss.item():.4f}")
    print(f"\n共享门控权重（第1个样本，所有任务共用）:")
    print(f"  专家0: {gate_weights[0][0].item():.3f}")
    print(f"  专家1: {gate_weights[0][1].item():.3f}")
    print(f"  专家2: {gate_weights[0][2].item():.3f}")
    print(f"  专家3: {gate_weights[0][3].item():.3f}")
    print(f"  → 只有一个门控，所有任务被迫用同一套权重")
    print(f"  → 任务需求冲突时无法调和，这就是MMOE要解决的问题")

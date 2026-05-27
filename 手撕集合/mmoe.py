import torch
import torch.nn as nn
import torch.nn.functional as F

class MMoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, num_tasks=2, expert_units=64, gate_units=32):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Experts（共享专家）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_units),
                nn.ReLU(),
                nn.Linear(expert_units, expert_units)
            ) for _ in range(num_experts)
        ])
        
        # Gates（每个任务一个门控）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_units),
                nn.ReLU(),
                nn.Linear(gate_units, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
        # Task-specific towers
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_units, gate_units),
                nn.ReLU(),
                nn.Linear(gate_units, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            task_outputs: List of (batch_size, 1), length = num_tasks
        """
        # Experts forward
        expert_outputs = [expert(x) for expert in self.experts]  # List of (batch_size, expert_units)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, expert_units)
        
        # Gates forward
        gate_outputs = []
        for i in range(self.num_tasks):
            gate_weight = self.gates[i](x)  # (batch_size, num_experts)
            gate_outputs.append(gate_weight.unsqueeze(-1))  # (batch_size, num_experts, 1)
        
        # Weighted sum of experts
        task_outputs = []
        for i in range(self.num_tasks):
            weighted_expert = (expert_outputs * gate_outputs[i]).sum(dim=1)  # (batch_size, expert_units)
            task_output = self.towers[i](weighted_expert)  # (batch_size, 1)
            task_outputs.append(task_output)
        
        return task_outputs

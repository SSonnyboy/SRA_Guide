"""
Shared-Bottom 多目标模型
========================
核心思想：所有目标共享一个底层网络，各自加一个任务塔
- 共享底层：提取通用特征（比如用户兴趣、物品属性）
- 任务塔：每个目标（点击率、点赞、收藏...）各自学一个

优点：
1. 结构简单，容易实现
2. 共享参数减少过拟合，数据利用率高
3. 训练快（共享部分只算一次）

缺点：
1. 所有任务被迫共享同一个底层表示
2. 如果任务之间相关性不高，会互相拖累（负迁移）
3. 底层网络容量有限，表达能力不足
"""

import torch
import torch.nn as nn

# ============================================================
# 模型定义
# ============================================================
class SharedBottom(nn.Module):
    def __init__(self, input_dim, shared_dims, task_dims, num_tasks):
        """
        input_dim:   输入特征维度
        shared_dims: 共享底层的隐藏层维度 [256, 128]
        task_dims:   每个任务塔的隐藏层维度 [64]
        num_tasks:   任务数量（比如点击、点赞、收藏 = 3个）
        """
        super().__init__()

        # ---- 共享底层：所有任务共用 ----
        shared_layers = []
        dim = input_dim
        for h in shared_dims:
            shared_layers.append(nn.Linear(dim, h))
            shared_layers.append(nn.ReLU())
            dim = h
        self.shared_bottom = nn.Sequential(*shared_layers)

        # ---- 任务塔：每个任务独立一个 ----
        self.task_towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower_layers = []
            tower_dim = shared_dims[-1]
            for h in task_dims:
                tower_layers.append(nn.Linear(tower_dim, h))
                tower_layers.append(nn.ReLU())
                tower_dim = h
            tower_layers.append(nn.Linear(tower_dim, 1))
            self.task_towers.append(nn.Sequential(*tower_layers))

    def forward(self, x):
        """
        x: (batch, input_dim) 输入特征
        返回: list of (batch, 1)，每个任务的预测
        """
        # 共享底层提取通用特征
        shared_feat = self.shared_bottom(x)

        # 每个任务塔独立预测
        outputs = []
        for tower in self.task_towers:
            outputs.append(torch.sigmoid(tower(shared_feat)))

        return outputs


# ============================================================
# 训练演示
# ============================================================
if __name__ == "__main__":
    input_dim = 32
    num_tasks = 3  # 点击率、点赞率、收藏率
    batch_size = 64

    model = SharedBottom(
        input_dim=input_dim,
        shared_dims=[256, 128],
        task_dims=[64],
        num_tasks=num_tasks
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # 模拟数据
    x = torch.randn(batch_size, input_dim)
    labels = [torch.randint(0, 2, (batch_size, 1)).float() for _ in range(num_tasks)]

    # 前向传播
    preds = model(x)

    # 多任务联合损失
    total_loss = 0
    for i in range(num_tasks):
        task_loss = loss_fn(preds[i], labels[i])
        total_loss += task_loss

    total_loss.backward()
    optimizer.step()

    print(f"总Loss: {total_loss.item():.4f}")
    for i in range(num_tasks):
        print(f"  任务{i} Loss: {loss_fn(preds[i], labels[i]).item():.4f}")
    print(f"\n模型结构:\n{model}")

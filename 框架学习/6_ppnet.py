"""
PPNet (Parameter Personalized Network)
========================================
核心思想：不同用户应该用不同的网络参数
- 传统模型：所有用户共享同一套权重 W
- PPNet：用用户特征生成个性化门控 gate，gate × 共享权重 = 该用户的专属参数

改进了谁？所有共享参数的模型（DeepFM/DCN/AutoInt 的 Deep 部分）
    传统模型：y = W·x，用户A和用户B用同一个W
    PPNet：y = (W × gate_user)·x，不同用户的gate不同

优点：
1. 真正的个性化：不同用户对特征的敏感度不同
2. 参数效率高：只多了一个轻量的Gate网络
3. 可解释性：gate值能看出每个用户关注哪些维度

缺点：
1. Gate网络依赖用户特征质量，冷启动用户效果差
2. 增加了计算量（多了一次前向传播）
3. 训练不稳定，需要调参
"""

import torch
import torch.nn as nn

# ============================================================
# PPNet 核心层：个性化全连接
# ============================================================
class PersonalizedLinear(nn.Module):
    """
    传统 Linear: y = W·x + b
    PPNet:       y = (W × gate(u))·x + b
                 gate(u) 是由用户特征生成的门控向量
    """
    def __init__(self, input_dim, output_dim, user_dim):
        super().__init__()
        # 基础权重（所有用户共享）
        self.base_weight = nn.Linear(input_dim, output_dim, bias=True)

        # Gate网络：用户特征 → 个性化门控
        self.gate_net = nn.Sequential(
            nn.Linear(user_dim, output_dim),
            nn.Sigmoid()   # 输出 0~1，控制每个输出维度的开关
        )

    def forward(self, x, user_feat):
        """
        x:         (batch, input_dim)   输入特征
        user_feat: (batch, user_dim)    用户特征（如用户ID的embedding）
        返回:      (batch, output_dim)  个性化后的输出
        """
        # 基础输出
        base_out = self.base_weight(x)    # (batch, output_dim)

        # 用户个性化门控
        gate = self.gate_net(user_feat)   # (batch, output_dim)

        # 个性化 = 基础 × 门控
        return base_out * gate


# ============================================================
# 完整的 PPNet-Deep 模块
# ============================================================
class PPNetDeep(nn.Module):
    """
    把传统MLP的每一层都替换成 PersonalizedLinear
    """
    def __init__(self, input_dim, hidden_dims, user_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(PersonalizedLinear(prev_dim, h_dim, user_dim))
            prev_dim = h_dim
        # 最后一层不需要个性化（输出是1维的logit）
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x, user_feat):
        for layer in self.layers:
            x = torch.relu(layer(x, user_feat))
        return self.output_layer(x)


# ============================================================
# 完整推荐模型：Embedding + PPNet
# ============================================================
class PPNetRanking(nn.Module):
    def __init__(self, num_fields, embed_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.num_fields = num_fields
        input_dim = num_fields * embed_dim
        user_dim = embed_dim  # 用户特征维度

        # 所有特征的Embedding
        self.embedding = nn.Embedding(50, embed_dim)

        # PPNet-Deep：每一层都个性化
        self.ppnet_deep = PPNetDeep(input_dim, hidden_dims, user_dim)

    def forward(self, x):
        """
        x: (batch, num_fields) 特征索引，第0个是用户ID
        """
        emb = self.embedding(x)  # (batch, num_fields, embed_dim)

        # 用户特征：取第0个特征（用户ID）的embedding
        user_feat = emb[:, 0, :]              # (batch, embed_dim)

        # 所有特征拼接
        all_feat = emb.view(emb.size(0), -1)  # (batch, num_fields * embed_dim)

        # PPNet预测
        logit = self.ppnet_deep(all_feat, user_feat)
        return torch.sigmoid(logit)


# ============================================================
# 对比实验：PPNet vs 传统共享参数
# ============================================================
class TraditionalDeep(nn.Module):
    """传统模型：所有用户共享参数"""
    def __init__(self, num_fields, embed_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.embedding = nn.Embedding(50, embed_dim)
        input_dim = num_fields * embed_dim
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.embedding(x)
        flat = emb.view(emb.size(0), -1)
        return torch.sigmoid(self.deep(flat))


if __name__ == "__main__":
    num_fields = 10
    batch_size = 32

    # ---- PPNet 模型 ----
    ppnet_model = PPNetRanking(num_fields=num_fields, embed_dim=8, hidden_dims=[64, 32])
    optimizer1 = torch.optim.Adam(ppnet_model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    x = torch.randint(0, 50, (batch_size, num_fields))
    y = torch.randint(0, 2, (batch_size, 1)).float()

    pred = ppnet_model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer1.step()

    print(f"PPNet Loss: {loss.item():.4f}")

    # ---- 传统模型 ----
    trad_model = TraditionalDeep(num_fields=num_fields, embed_dim=8, hidden_dims=[64, 32])
    optimizer2 = torch.optim.Adam(trad_model.parameters(), lr=0.001)

    pred = trad_model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer2.step()

    print(f"传统模型 Loss: {loss.item():.4f}")

    # ---- 展示PPNet的个性化效果 ----
    print("\n=== 个性化门控演示 ===")
    with torch.no_grad():
        # 不同用户，同样的商品特征
        # 用户1（重度用户）和用户2（新用户）
        item_features = torch.randint(0, 50, (2, num_fields))
        item_features[0, 0] = 1   # 用户1
        item_features[1, 0] = 2   # 用户2

        emb = ppnet_model.embedding(item_features)
        user1_feat = emb[0:1, 0, :]   # 用户1的embedding
        user2_feat = emb[1:2, 0, :]   # 用户2的embedding

        # 查看第一个PPNet层的gate值
        first_layer = ppnet_model.ppnet_deep.layers[0]
        gate1 = first_layer.gate_net(user1_feat)
        gate2 = first_layer.gate_net(user2_feat)

        print(f"用户1的gate: {gate1[0].numpy().round(3)}")
        print(f"用户2的gate: {gate2[0].numpy().round(3)}")
        print(f"gate不同 → 两个用户对同一维度的敏感度不同 → 真正的个性化")

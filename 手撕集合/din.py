import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, hidden_units=[64, 32]):
        super(DIN, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Linear(user_feature_dim, hidden_units[0])
        self.item_embedding = nn.Linear(item_feature_dim, hidden_units[0])
        
        # Activation Unit (计算注意力权重)
        self.attention_net = nn.Sequential(
            nn.Linear(4 * hidden_units[0], hidden_units[1]),
            nn.PReLU(),
            nn.Linear(hidden_units[1], 1, bias=False),  # 输出注意力分数（无 Softmax）
        )
        
        # Final MLP
        self.fc = nn.Sequential(
            nn.Linear(3 * hidden_units[0], hidden_units[1]),
            nn.PReLU(),
            nn.Linear(hidden_units[1], 1),
            nn.Sigmoid()  # 输出 CTR 概率
        )

    def forward(self, user_features, item_features, user_behavior_seq):
        """
        Args:
            user_features: (batch_size, user_feature_dim)
            item_features: (batch_size, item_feature_dim)
            user_behavior_seq: (batch_size, seq_len, item_feature_dim)
        Returns:
            ctr_prob: (batch_size, 1)
        """
        # Embedding
        user_emb = self.user_embedding(user_features)  # (batch_size, hidden_units[0])
        item_emb = self.item_embedding(item_features)  # (batch_size, hidden_units[0])
        behavior_emb = self.item_embedding(user_behavior_seq)  # (batch_size, seq_len, hidden_units[0])

        # Attention weights
        seq_len = behavior_emb.size(1)
        item_emb_expanded = item_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_units[0])
        
        # 计算 Query-Key 交互特征
        concat_input = torch.cat([
            item_emb_expanded, behavior_emb,
            item_emb_expanded - behavior_emb,
            item_emb_expanded * behavior_emb  # 哈达玛积（Hadamard Product）
        ], dim=-1)  # (batch_size, seq_len, 4 * hidden_units[0])
        
        # 计算注意力分数（无 Softmax）
        attention_scores = self.attention_net(concat_input)  # (batch_size, seq_len, 1)
        
        # 加权求和
        weighted_behavior = (attention_scores * behavior_emb).sum(dim=1)  # (batch_size, hidden_units[0])
        
        # 拼接用户、候选商品、加权后的行为
        final_input = torch.cat([user_emb, item_emb, weighted_behavior], dim=-1)
        
        # 预测 CTR
        ctr_prob = self.fc(final_input)
        return ctr_prob

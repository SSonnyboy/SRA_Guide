import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) 实现
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头总数
        n_groups: KV 头的分组数量 (n_groups <= n_heads)
        dropout: dropout 概率
    """
    def __init__(self, d_model, n_heads, n_groups, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = d_model // n_heads
        self.kv_heads_per_group = n_heads // n_groups  # 每组包含的KV头数
        
        # Q 投影：每个头都有独立的Q
        self.w_q = nn.Linear(d_model, d_model)
        
        # K, V 投影：每组共享K和V
        self.w_k = nn.Linear(d_model, self.head_dim * n_groups)
        self.w_v = nn.Linear(d_model, self.head_dim * n_groups)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            past_kv: 过去的KV缓存 (k_cache, v_cache)
            use_cache: 是否使用KV缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 生成查询 Q
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 生成键 K 和值 V
        k = self.w_k(x)  # [batch_size, seq_len, head_dim * n_groups]
        v = self.w_v(x)  # [batch_size, seq_len, head_dim * n_groups]
        
        # 重塑K和V以匹配分组结构
        k = k.view(batch_size, seq_len, self.n_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_groups, self.head_dim).transpose(1, 2)
        # [batch_size, n_groups, seq_len, head_dim]
        
        # 处理KV缓存（用于自回归生成）
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # 保存当前的KV用于缓存
        present_kv = (k, v) if use_cache else None
        
        # 复制K和V以匹配Q的头数
        # 每个KV组需要重复 kv_heads_per_group 次
        k = k.repeat_interleave(self.kv_heads_per_group, dim=1)
        v = v.repeat_interleave(self.kv_heads_per_group, dim=1)
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch_size, n_heads, seq_len, seq_len]
        
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算上下文向量
        context = torch.matmul(attn_weights, v)
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 重塑并投影输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output, attn_weights, present_kv
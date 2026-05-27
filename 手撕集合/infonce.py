import torch
import torch.nn.functional as F

def inbatch_infonce(query_emb, pos_emb, temperature=0.1):
    """
    In-batch InfoNCE loss implementation
    
    Args:
        query_emb: [batch_size, dim] 查询向量
        pos_emb: [batch_size, dim] 正样本向量
        neg_emb: [batch_size, num_neg, dim ] 显式负样本向量（可选）
        temperature: float 温度系数
    Returns:
        loss: scalar
    """
    batch_size = query_emb.size(0)
    
    # 计算查询与所有样本的相似度（包括in-batch负样本）
    # [batch_size, batch_size]
    all_sim = torch.mm(query_emb, pos_emb.T) / temperature
    
    # 对角线是正样本
    diag_mask = torch.eye(batch_size, device=query_emb.device).bool()
    
    # 计算log_softmax（数值稳定实现）
    logits = all_sim - all_sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    
    # 只选择正样本位置的概率
    pos_log_prob = log_prob[diag_mask].view(batch_size)
    
    # 损失是负的log似然
    loss = -pos_log_prob.mean()
    
    return loss

# 使用示例
batch_size = 32
dim = 128
query = torch.randn(batch_size, dim)
positive = torch.randn(batch_size, dim)

loss = inbatch_infonce(query, positive, temperature=0.1)
print(loss)
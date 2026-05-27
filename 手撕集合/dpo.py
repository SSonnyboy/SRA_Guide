import torch
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta=0.1):
    """
    Direct Preference Optimization (DPO) 损失函数
    
    参数:
        pi_logps: 策略模型对所有response的对数概率分布 (shape: (batch_size, num_responses))
        ref_logps: 参考模型对所有response的对数概率分布 (shape: (batch_size, num_responses))
        yw_idxs: 每个样本中优选回答的索引 (shape: (batch_size,))
        yl_idxs: 每个样本中劣选回答的索引 (shape: (batch_size,))
        
    返回:
        loss: DPO损失值
        metrics: 包含有用指标的字典
    """
    # 应该沿着response维度（第1维）进行gather
    pi_yw_logps = pi_logps.gather(1, yw_idxs.unsqueeze(1)).squeeze(1)
    pi_yl_logps = pi_logps.gather(1, yl_idxs.unsqueeze(1)).squeeze(1)

    ref_yw_logps = ref_logps.gather(1, yw_idxs.unsqueeze(1)).squeeze(1)
    ref_yl_logps = ref_logps.gather(1, yl_idxs.unsqueeze(1)).squeeze(1)
    
    # 计算策略模型和参考模型的对数概率差
    logits = pi_yw_logps - pi_yl_logps
    ref_logits = ref_yw_logps - ref_yl_logps
    
    # 计算DPO损失
    losses = -F.logsigmoid(beta * (logits - ref_logits))
    
    # 计算奖励准确率（用于监控）
    reward_acc = (logits > ref_logits).float().mean()
    
    # 返回平均损失和指标
    return losses.mean(), {
        "reward_acc": reward_acc,
        "logits/yw": pi_yw_logps.mean(),
        "logits/yl": pi_yl_logps.mean(),
        "logits/yw_ref": ref_yw_logps.mean(),
        "logits/yl_ref": ref_yl_logps.mean(),
    }

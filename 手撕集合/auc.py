import numpy as np

# 精确版
def auc_rank(y_true, y_pred):
    # 确保输入是 numpy 数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 获取正负样本的索引
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)  # 正样本数
    n = len(neg_idx)  # 负样本数
    
    if m == 0 or n == 0:
        return 0.5  # 如果只有一类，AUC=0.5（随机）
    
    # 对所有样本的预测值进行排名（升序，排名从1开始）
    ranks = np.argsort(np.argsort(y_pred)) + 1
    
    # 计算正样本的排名之和
    sum_rank_pos = np.sum(ranks[pos_idx])
    
    # 计算AUC
    auc = (sum_rank_pos - m * (m + 1) / 2) / (m * n)
    
    return auc

# 近似版
import numpy as np
def auc(y_true, y_score):
    # 1. 排序
    idx = np.argsort(y_score)[::-1]
    y = y_true[idx]
    
    # 2. 算PN
    P = np.sum(y==1)
    N = len(y)-P
    
    FP=TP=auc=0
    pre_fpr=0
    
    # 3. 遍历
    for i in y:
        if i==1: TP+=1
        else: FP+=1
        
        cur_fpr = FP/N
        cur_tpr = TP/P
        auc += (cur_fpr-pre_fpr)*cur_tpr
        pre_fpr = cur_fpr
    
    auc += (1-pre_fpr)*1
    return auc
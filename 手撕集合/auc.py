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
def auc_rectangle(y_true, y_scores):
    # 按预测得分降序排序
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # 统计正负样本数
    P = np.sum(y_true == 1)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return 0.5  # 全正或全负时AUC无意义

    # 初始化变量
    FP, TP = 0, 0
    prev_fpr, prev_tpr = 0, 0
    auc = 0.0

    # 遍历所有可能的阈值
    for i in range(len(y_scores_sorted)):
        if y_true_sorted[i] == 1:
            TP += 1
        else:
            FP += 1
        curr_fpr = FP / N
        curr_tpr = TP / P
        # 计算当前矩形的面积（宽 × 高）
        auc += (curr_fpr - prev_fpr) * curr_tpr
        prev_fpr, prev_tpr = curr_fpr, curr_tpr

    # 处理最后一个矩形（阈值≤最小得分）
    auc += (1 - prev_fpr) * 1  # 最后一个点(FPR=1, TPR=1)
    return auc
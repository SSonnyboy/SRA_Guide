import numpy as np


def sigmoid(x):
    # 考虑数值稳定： x<=0时 出现问题 
    # 使用分段思想
    out = np.where(
        x>=0,
        1/(1+np.exp(-x)),
        np.exp(x)/(1+np.exp(x))
    )
    return out


def softmax(x):
    x_shifted = x - np.max(x, arxis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def bce(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    out = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )
    return out

def ce(y_true, y_pred_logits):
    # softmax
    y_pred_soft = softmax(y_pred_logits)
    n = y_true.shape[0]
    y_entropy = y_pred_soft[np.arange(n), y_true]
    return -np.mean(np.log(y_entropy))


def bn(gamma, beta, x):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + 1e-5)
    return gamma * x_norm + beta

def ln(gamma, beta, x):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + 1e-5)
    return gamma * x_norm + beta


def auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    if n==0 or m==0:
        return 0.5
    ranks = np.argsort(np.argsort(y_score)) + 1
    sum_rank_pos = np.sum(ranks[pos_idx])
    auc = (sum_rank_pos - m * (m + 1) / 2) / (m * n)
    return auc

def auc_roc(y_true, y_score):
    idx = np.argsort(y_score)[::-1]
    y_true = y_true[idx]
    P = np.sum(y_true==1)
    N = len(y_true)-P
    FP=TP=auc=0
    pre_fpr=0
    for i in y_true:
        if i==1: TP+=1
        else: FP+=1
        cur_fpr = FP/N
        cur_tpr = TP/P
        auc += (cur_fpr-pre_fpr)*cur_tpr
        pre_fpr = cur_fpr
    auc += (1-pre_fpr)*1
    return auc

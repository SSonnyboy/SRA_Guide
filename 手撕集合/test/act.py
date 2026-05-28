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



class DNN:
    def __init__(self, indim, hiddim, outdim=1, lr=0.01):
        self.lr = lr
        self.w1 = np.random.randn(indim, hiddim)* 0.01
        self.b1 = np.zeros((1, hiddim))

        self.w2 = np.random.randn(hiddim, outdim)* 0.01
        self.b2 = np.zeros((1, outdim))

    @staticmethod
    def sigmoid(x):
        out = np.where(
            x>=0,
            1/(1+np.exp(-x)),
            np.exp(x)/(1+np.exp(x))
        )
        return out
    
    @staticmethod
    def relu(x):
        return np.where(x>0, x, 0)
    
    @staticmethod
    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def forward(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)
        cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2, cache
    
    def backward(self, X, Y, cache):
        z1, a1, z2, a2 = cache["z1"], cache["a1"], cache["z2"], cache["a2"]
        m = X.shape[0]

        dz2 = a2 - Y
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.w2.T)
        dz1 = self.relu_backward(da1, z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2    
        self.b2 -= self.lr * db2
    
    def step(self, X, Y):
        a2, cache = self.forward(X)
        self.backward(X, Y, cache)
        return a2





import torch
import math
from torch import nn
from torch.nn import functional as F

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        return output
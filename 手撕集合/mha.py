import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
        
        self.k_cache = None
        self.v_cache = None
    
    def softmax(self, x, axis=-1):
        e_x = np.exp(x-np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        Q = Q.reshape(batch_size, 1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, 1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, 1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        if self.k_cache is not None:
            K = np.concatenate([self.k_cache, K],axis=2)
            V = np.concatenate([self.v_cache, V],axis=2)
        self.k_cache = K
        self.v_cache = V
        
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn = self.softmax(scores, axis=-1)
        context = np.matmul(attn, V)
        
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, 1, self.d_model)
        output = np.dot(context, self.W_o)
        return output
import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    # 初始化中心点（从数据中随机选 k 个）
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # 计算每个点到中心的距离，分配类别
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # (N, k)
        labels = np.argmin(distances, axis=1)  # (N,)
        
        # 更新中心点
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 判断是否收敛
        if np.linalg.norm(new_centers - centers) < tol:
            print(f"Converged at iteration {i+1}")
            break
            
        centers = new_centers
    
    return labels, centers

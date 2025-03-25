import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import silhouette_score

class KMeansParams:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

class KMeans:
    def __init__(self, params=None):
        if params is None:
            params = KMeansParams()
        self.n_clusters = params.n_clusters
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.silhouette_score = None

    def _initialize_centroids(self, X):
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[indices].clone()

    def fit(self, X):
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 初始化聚类中心
        self._initialize_centroids(X_tensor)
        self.centroids = self.centroids.to(self.device)

        prev_centroids = torch.zeros_like(self.centroids)
        history = {
            'inertia': [],
            'centroid_shifts': [],
            'silhouette_scores': []
        }

        for _ in range(self.max_iter):
            # 计算每个样本到各个聚类中心的距离
            distances = torch.cdist(X_tensor, self.centroids)
            
            # 为每个样本分配最近的聚类中心
            self.labels = torch.argmin(distances, dim=1)
            
            # 保存旧的聚类中心
            prev_centroids.copy_(self.centroids)
            
            # 更新聚类中心
            for i in range(self.n_clusters):
                mask = self.labels == i
                if mask.sum() > 0:
                    self.centroids[i] = X_tensor[mask].mean(dim=0)
            
            # 计算聚类中心的移动距离
            centroid_shift = torch.norm(self.centroids - prev_centroids)
            
            # 计算惯性（样本到其聚类中心的距离平方和）
            self.inertia = torch.sum(torch.min(distances, dim=1)[0])
            
            # 计算轮廓系数
            current_labels = self.labels.cpu().numpy()
            if len(np.unique(current_labels)) > 1:  # 确保至少有两个类别
                self.silhouette_score = silhouette_score(X, current_labels)
            else:
                self.silhouette_score = 0
            
            # 记录历史
            history['inertia'].append(self.inertia.item())
            history['centroid_shifts'].append(centroid_shift.item())
            history['silhouette_scores'].append(self.silhouette_score)
            
            # 如果聚类中心基本不再移动，则停止迭代
            if centroid_shift < self.tol:
                break

        # 将结果转移到CPU并转换为NumPy数组
        self.labels = self.labels.cpu().numpy()
        self.centroids = self.centroids.cpu().numpy()
        self.inertia = self.inertia.cpu().item()

        # 清理GPU内存
        del X_tensor
        del distances
        del prev_centroids

        return history

    def predict(self, X):
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        centroids_tensor = torch.tensor(self.centroids, dtype=torch.float32).to(self.device)
        
        # 计算距离并分配标签
        distances = torch.cdist(X_tensor, centroids_tensor)
        labels = torch.argmin(distances, dim=1)
        
        # 清理GPU内存
        del X_tensor
        del centroids_tensor
        del distances
        
        return labels.cpu().numpy()

    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'inertia': self.inertia,
            'centroids': self.centroids.tolist() if self.centroids is not None else None
        }

    def dispose(self):
        self.centroids = None
        self.labels = None
        self.inertia = None
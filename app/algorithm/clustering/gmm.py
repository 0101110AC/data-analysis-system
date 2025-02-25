import torch
import torch.nn as nn
import numpy as np

class GMMParams:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3, covariance_type='full'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type  # 'full', 'tied', 'diag', 'spherical'

class GMM:
    def __init__(self, params=None):
        if params is None:
            params = GMMParams()
        self.n_components = params.n_components
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.covariance_type = params.covariance_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 模型参数
        self.weights = None  # 混合权重
        self.means = None    # 均值
        self.covs = None     # 协方差矩阵
        self.labels = None   # 聚类标签

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        # 随机初始化权重
        self.weights = torch.ones(self.n_components, device=self.device) / self.n_components
        
        # 使用K-means++初始化均值
        centroids = torch.zeros((self.n_components, n_features), device=self.device)
        # 随机选择第一个中心点
        first_centroid = X[torch.randint(n_samples, (1,))]
        centroids[0] = first_centroid
        
        # 选择剩余的中心点
        for i in range(1, self.n_components):
            # 计算到最近中心点的距离
            distances = torch.cdist(X, centroids[:i])
            min_distances = torch.min(distances, dim=1)[0]
            # 按概率选择下一个中心点
            probs = min_distances / min_distances.sum()
            next_centroid_idx = torch.multinomial(probs, 1)
            centroids[i] = X[next_centroid_idx]
        
        self.means = centroids
        
        # 初始化协方差矩阵
        if self.covariance_type == 'spherical':
            self.covs = torch.eye(n_features, device=self.device).unsqueeze(0).repeat(self.n_components, 1, 1)
        elif self.covariance_type == 'diag':
            self.covs = torch.eye(n_features, device=self.device).unsqueeze(0).repeat(self.n_components, 1, 1)
        elif self.covariance_type == 'tied':
            self.covs = torch.eye(n_features, device=self.device).unsqueeze(0).repeat(self.n_components, 1, 1)
        else:  # 'full'
            self.covs = torch.stack([torch.eye(n_features, device=self.device) for _ in range(self.n_components)])
        
        # 添加小的对角项以确保数值稳定性
        self.covs += torch.eye(n_features, device=self.device).unsqueeze(0) * 1e-6

    def _e_step(self, X):
        n_samples = X.shape[0]
        
        # 计算每个样本属于每个组件的概率
        resp = torch.zeros((n_samples, self.n_components), device=self.device)
        
        for k in range(self.n_components):
            # 计算多元正态分布概率密度
            diff = X - self.means[k]
            inv_cov = torch.inverse(self.covs[k])
            mahalanobis = torch.sum((diff @ inv_cov) * diff, dim=1)
            det_cov = torch.det(self.covs[k])
            norm_const = 1.0 / torch.sqrt((2 * torch.pi) ** X.shape[1] * det_cov)
            resp[:, k] = self.weights[k] * norm_const * torch.exp(-0.5 * mahalanobis)
        
        # 归一化响应
        total_resp = torch.sum(resp, dim=1, keepdim=True)
        resp = resp / total_resp
        
        return resp

    def _m_step(self, X, resp):
        n_samples = X.shape[0]
        
        # 更新权重
        nk = torch.sum(resp, dim=0)
        self.weights = nk / n_samples
        
        # 更新均值
        self.means = torch.matmul(resp.T, X) / nk.unsqueeze(1)
        
        # 更新协方差矩阵
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = (diff.T @ (resp[:, k:k+1] * diff)) / nk[k]
            # 添加小的对角项以确保数值稳定性
            self.covs[k] += torch.eye(X.shape[1], device=self.device) * 1e-6

    def fit(self, X):
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 初始化参数
        self._initialize_parameters(X_tensor)
        
        history = {
            'log_likelihood': [],
            'n_iter': 0,
            'bic': [],
            'aic': []
        }
        
        prev_log_likelihood = -float('inf')
        
        for iteration in range(self.max_iter):
            # E步：计算响应
            resp = self._e_step(X_tensor)
            
            # 计算对数似然
            log_likelihood = torch.sum(torch.log(torch.sum(resp, dim=1)))
            history['log_likelihood'].append(log_likelihood.item())
            
            # 计算BIC和AIC
            n_samples = X_tensor.shape[0]
            n_features = X_tensor.shape[1]
            n_parameters = self.n_components * (1 + n_features + n_features * n_features)
            bic = -2 * log_likelihood.item() + n_parameters * torch.log(torch.tensor(n_samples)).item()
            aic = -2 * log_likelihood.item() + 2 * n_parameters
            
            history['bic'].append(bic)
            history['aic'].append(aic)
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            # M步：更新参数
            self._m_step(X_tensor, resp)
            
            prev_log_likelihood = log_likelihood
            history['n_iter'] += 1
        
        # 保存聚类标签
        self.labels = torch.argmax(resp, dim=1).cpu().numpy()
        
        # 清理GPU内存
        del X_tensor
        del resp
        
        return history

    def predict(self, X):
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 计算响应并返回最可能的类别
        resp = self._e_step(X_tensor)
        labels = torch.argmax(resp, dim=1)
        
        # 清理GPU内存
        del X_tensor
        del resp
        
        return labels.cpu().numpy()

    def get_params(self):
        return {
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'weights': self.weights.cpu().numpy().tolist() if self.weights is not None else None,
            'means': self.means.cpu().numpy().tolist() if self.means is not None else None
        }

    def dispose(self):
        self.weights = None
        self.means = None
        self.covs = None
        self.labels = None
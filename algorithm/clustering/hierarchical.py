import torch
import numpy as np

class HierarchicalParams:
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage  # 'ward', 'complete', 'average', 'single'

class Hierarchical:
    def __init__(self, params=None):
        if params is None:
            params = HierarchicalParams()
        self.n_clusters = params.n_clusters
        self.linkage = params.linkage
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = None

    def _compute_distances(self, X):
        n_samples = X.shape[0]
        distances = torch.cdist(X, X)
        # 将对角线设置为无穷大，避免选择自身
        distances.fill_diagonal_(float('inf'))
        return distances

    def _merge_clusters(self, distances, labels, i, j):
        # 将j类的所有点合并到i类
        mask_j = labels == j
        labels[mask_j] = i
        # 更新大于j的标签，保持连续性
        labels[labels > j] -= 1
        return labels

    def _update_distances(self, distances, i, j, labels, X):
        if self.linkage == 'single':
            # 单链接：最小距离
            for k in range(len(distances)):
                if k != i and k != j:
                    distances[i,k] = distances[k,i] = min(distances[i,k], distances[j,k])
        elif self.linkage == 'complete':
            # 全链接：最大距离
            for k in range(len(distances)):
                if k != i and k != j:
                    distances[i,k] = distances[k,i] = max(distances[i,k], distances[j,k])
        elif self.linkage == 'average':
            # 平均链接：平均距离
            for k in range(len(distances)):
                if k != i and k != j:
                    ni = torch.sum(labels == i)
                    nj = torch.sum(labels == j)
                    distances[i,k] = distances[k,i] = (ni * distances[i,k] + nj * distances[j,k]) / (ni + nj)
        elif self.linkage == 'ward':
            # Ward链接：最小化类内方差
            cluster_i = X[labels == i]
            cluster_j = X[labels == j]
            merged_cluster = torch.cat([cluster_i, cluster_j])
            centroid = torch.mean(merged_cluster, dim=0)
            for k in range(len(distances)):
                if k != i and k != j:
                    cluster_k = X[labels == k]
                    ward_dist = torch.sum((cluster_k - centroid) ** 2)
                    distances[i,k] = distances[k,i] = ward_dist

        # 删除j行和j列
        distances = torch.cat([distances[:j], distances[j+1:]])
        distances = torch.cat([distances[:,:j], distances[:,j+1:]], dim=1)
        return distances

    def fit(self, X):
        # 保存训练数据
        self.X_train = X
        
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        n_samples = X_tensor.shape[0]
        
        # 初始化标签：每个样本一个类
        self.labels = torch.arange(n_samples, device=self.device)
        
        # 计算初始距离矩阵
        distances = self._compute_distances(X_tensor)
        original_distances = distances.clone()
        
        history = {
            'n_clusters': [],
            'merged_clusters': [],
            'cophenetic_matrix': None
        }
        
        # 初始化共病矩阵
        cophenetic_matrix = torch.zeros_like(distances)
        current_height = 0
        
        n_clusters = n_samples
        while n_clusters > self.n_clusters:
            # 找到最近的两个类
            min_dist = torch.min(distances)
            indices = torch.where(distances == min_dist)
            i, j = indices[0][0], indices[1][0]  # 获取第一个最小距离对应的索引
            i, j = min(i.item(), j.item()), max(i.item(), j.item())  # 确保i < j
            
            # 更新共病矩阵
            mask_i = self.labels == i
            mask_j = self.labels == j
            cophenetic_matrix[mask_i][:, mask_i] = current_height
            cophenetic_matrix[mask_j][:, mask_j] = current_height
            cophenetic_matrix[mask_i][:, mask_j] = current_height
            cophenetic_matrix[mask_j][:, mask_i] = current_height
            current_height += min_dist.item()
            
            # 记录合并历史
            history['merged_clusters'].append((i, j))
            
            # 合并类
            self.labels = self._merge_clusters(distances, self.labels, i, j)
            
            # 更新距离矩阵
            distances = self._update_distances(distances, i, j, self.labels, X_tensor)
            
            n_clusters -= 1
            history['n_clusters'].append(n_clusters)
        
        # 计算Cophenetic相关系数
        # 获取上三角矩阵的索引（不包括对角线）
        indices = torch.triu_indices(n_samples, n_samples, offset=1)
        original_dist_values = original_distances[indices[0], indices[1]]
        cophenetic_values = cophenetic_matrix[indices[0], indices[1]]
        
        # 计算相关系数
        mean_orig = torch.mean(original_dist_values)
        mean_coph = torch.mean(cophenetic_values)
        
        numerator = torch.sum((original_dist_values - mean_orig) * (cophenetic_values - mean_coph))
        denominator = torch.sqrt(torch.sum((original_dist_values - mean_orig) ** 2) * 
                                torch.sum((cophenetic_values - mean_coph) ** 2))
        
        if denominator > 0:
            cophenetic_corr = (numerator / denominator).item()
        else:
            cophenetic_corr = 0.0
        history['cophenetic_correlation'] = cophenetic_corr
        history['cophenetic_matrix'] = cophenetic_matrix.cpu().numpy()
        
        # 将结果转移到CPU
        self.labels = self.labels.cpu().numpy()
        
        # 清理GPU内存
        del X_tensor
        del distances
        del original_distances
        del cophenetic_matrix
        
        return history

    def predict(self, X):
        if self.labels is None:
            raise Exception("Model not fitted yet!")
            
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        centroids_tensor = torch.zeros((self.n_clusters, X.shape[1]), device=self.device)
        
        # 计算每个类的质心
        unique_labels = np.unique(self.labels)
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            if mask.any():
                centroids_tensor[i] = torch.mean(torch.tensor(self.X_train[mask], dtype=torch.float32).to(self.device), dim=0)
        
        # 为新数据点分配最近的类
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
            'linkage': self.linkage
        }

    def dispose(self):
        self.labels = None
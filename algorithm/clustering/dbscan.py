import torch
import numpy as np

class DBSCANParams:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # 邻域半径
        self.min_samples = min_samples  # 最小样本数

class DBSCAN:
    def __init__(self, params=None):
        if params is None:
            params = DBSCANParams()
        self.eps = params.eps
        self.min_samples = params.min_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = None

    def _find_neighbors(self, X, point_idx):
        # 计算所有点到当前点的距离
        distances = torch.norm(X - X[point_idx], dim=1)
        # 返回距离小于eps的点的索引
        return torch.where(distances <= self.eps)[0]

    def fit(self, X):
        # 保存训练数据用于后续预测
        self.X_train = X
        
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        n_samples = X_tensor.shape[0]
        self.labels = torch.full((n_samples,), -1, device=self.device)  # -1表示未分类

        cluster_id = 0
        history = {
            'n_clusters': [],
            'noise_points': [],
            'density_distribution': [],
            'noise_ratio': []
        }

        # 遍历所有点
        for point_idx in range(n_samples):
            if self.labels[point_idx] != -1:
                continue

            neighbors = self._find_neighbors(X_tensor, point_idx)

            # 如果邻域内点的数量小于min_samples，标记为噪声点
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -2  # -2表示噪声点
                continue

            # 开始一个新的聚类
            cluster_id += 1
            self.labels[point_idx] = cluster_id

            # 扩展聚类
            seeds = neighbors.tolist()
            for seed in seeds:
                if self.labels[seed] == -2:  # 噪声点可以成为边界点
                    self.labels[seed] = cluster_id
                if self.labels[seed] != -1:
                    continue

                self.labels[seed] = cluster_id
                new_neighbors = self._find_neighbors(X_tensor, seed)

                if len(new_neighbors) >= self.min_samples:
                    seeds.extend(new_neighbors.tolist())

            # 记录历史
            n_clusters = len(torch.unique(self.labels[self.labels >= 0]))
            n_noise = torch.sum(self.labels == -2).item()
            n_total = len(self.labels)
            
            # 计算每个簇的密度（每个簇中的点数除以总点数）
            density_dist = []
            for i in range(1, n_clusters + 1):
                cluster_size = torch.sum(self.labels == i).item()
                density = cluster_size / n_total
                density_dist.append(density)
            
            # 计算噪声点比例
            noise_ratio = n_noise / n_total
            
            history['n_clusters'].append(n_clusters)
            history['noise_points'].append(n_noise)
            history['density_distribution'].append(density_dist)
            history['noise_ratio'].append(noise_ratio)

        # 将结果转移到CPU
        self.labels = self.labels.cpu().numpy()

        # 清理GPU内存
        del X_tensor

        return history

    def predict(self, X):
        # 将输入数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        n_samples = X_tensor.shape[0]
        labels = torch.full((n_samples,), -2, device=self.device)  # 初始化为噪声点

        # 对每个新样本
        for i in range(n_samples):
            # 找到训练数据中的邻域点
            distances = torch.norm(torch.tensor(self.X_train, dtype=torch.float32).to(self.device) - X_tensor[i], dim=1)
            neighbors = torch.where(distances <= self.eps)[0]

            # 如果邻域内有足够的核心点，则分配到对应的类
            if len(neighbors) >= self.min_samples:
                # 找到邻域内的非噪声点标签
                neighbor_labels = self.labels[neighbors]
                valid_labels = neighbor_labels[neighbor_labels >= 0]
                
                if len(valid_labels) > 0:
                    # 分配最常见的类别
                    labels[i] = torch.mode(torch.tensor(valid_labels))[0]

        # 清理GPU内存
        del X_tensor
        
        return labels.cpu().numpy()

    def get_params(self):
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'n_clusters': len(np.unique(self.labels[self.labels >= 0])) if self.labels is not None else 0
        }

    def dispose(self):
        self.labels = None
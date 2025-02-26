import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import sqlite3

class DatasetManager:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, 'datasets.db')
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库，创建必要的表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建数据集元信息表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            type TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    
    def save_dataset(self, name, data, labels=None, dataset_type='generated', description=''):
        """保存数据集到CSV文件并记录元信息"""
        # 确保目录存在
        dataset_dir = os.path.join(self.base_dir, dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 保存数据到CSV
        df = pd.DataFrame(data)
        if labels is not None:
            df['label'] = labels
        
        file_path = os.path.join(dataset_dir, f'{name}.csv')
        df.to_csv(file_path, index=False)
        
        # 记录元信息到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO dataset_meta (name, type, description)
        VALUES (?, ?, ?)
        """, (name, dataset_type, description))
        
        conn.commit()
        conn.close()
    
    def load_dataset(self, name, dataset_type='generated'):
        """从CSV文件加载数据集"""
        file_path = os.path.join(self.base_dir, dataset_type, f'{name}.csv')
        df = pd.read_csv(file_path)
        
        if 'label' in df.columns:
            return df.drop('label', axis=1).values, df['label'].values
        return df.values, None
    
    def get_dataset_info(self, name):
        """获取数据集的元信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dataset_meta WHERE name = ?", (name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'type': result[2],
                'description': result[3],
                'created_at': result[4]
            }
        return None
    
    def generate_clustering_datasets(self):
        """生成聚类算法使用的数据集"""
        n_samples = 1000
        random_state = 42
        
        # 生成K-means数据集
        kmeans_X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                                cluster_std=[0.5, 0.8, 0.3], random_state=random_state)
        self.save_dataset('kmeans_data', kmeans_X, dataset_type='clustering',
                         description='Spherical clusters for K-means')
        
        # 生成DBSCAN数据集
        dbscan_X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
        self.save_dataset('dbscan_data', dbscan_X, dataset_type='clustering',
                         description='Non-spherical clusters for DBSCAN')
        
        # 生成层次聚类数据集
        hierarchical_X, _ = make_circles(n_samples=n_samples, noise=0.03, factor=0.5,
                                       random_state=random_state)
        self.save_dataset('hierarchical_data', hierarchical_X, dataset_type='clustering',
                         description='Concentric circles for hierarchical clustering')
        
        # 生成GMM数据集
        gmm_X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                             cluster_std=[0.5, 0.8, 1.0], random_state=random_state)
        self.save_dataset('gmm_data', gmm_X, dataset_type='clustering',
                         description='Gaussian mixture data for GMM')
    
    def generate_classification_datasets(self):
        """生成分类算法使用的数据集"""
        n_samples = 1000
        random_state = 42
        
        # 生成二分类数据集
        X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
        self.save_dataset('binary_classification', X, y, dataset_type='classification',
                         description='二分类月牙形数据集')
        
        # 生成多分类数据集
        X, y = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                         cluster_std=[1.0, 1.5, 0.5], random_state=random_state)
        self.save_dataset('multi_classification', X, y, dataset_type='classification',
                         description='多分类数据集')
        
        # 生成非线性分类数据集
        X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5,
                          random_state=random_state)
        self.save_dataset('nonlinear_classification', X, y, dataset_type='classification',
                         description='非线性分类数据集（同心圆）')
    
    def generate_regression_datasets(self):
        """生成回归算法使用的数据集"""
        from sklearn.datasets import fetch_california_housing, load_boston
        
        # 加载加利福尼亚房价数据集
        california = fetch_california_housing()
        X_california = california.data
        y_california = california.target
        self.save_dataset('california_housing', X_california, y_california,
                         dataset_type='regression',
                         description='加利福尼亚房价数据集，包含8个特征')
        
        # 生成简单的线性回归数据集
        n_samples = 1000
        X_linear = np.random.randn(n_samples, 3)
        y_linear = 2 * X_linear[:, 0] + 0.5 * X_linear[:, 1] - 1.5 * X_linear[:, 2] + np.random.randn(n_samples) * 0.1
        self.save_dataset('linear_regression', X_linear, y_linear,
                         dataset_type='regression',
                         description='简单的线性回归数据集，包含3个特征')
        
        # 生成非线性回归数据集
        X_nonlinear = np.random.randn(n_samples, 2)
        y_nonlinear = np.sin(X_nonlinear[:, 0]) + np.cos(X_nonlinear[:, 1]) + np.random.randn(n_samples) * 0.1
        self.save_dataset('nonlinear_regression', X_nonlinear, y_nonlinear,
                         dataset_type='regression',
                         description='非线性回归数据集，包含正弦和余弦特征')

    def load_mnist_data(self, max_samples=5000):
        """加载MNIST数据集并保存"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
        
        train_data = []
        train_labels = []
        sample_count = 0
        
        for data, label in train_loader:
            if sample_count >= max_samples:
                break
            train_data.append(data.numpy().reshape(-1, 784))
            train_labels.append(label.numpy())
            sample_count += len(data)
        
        X = np.concatenate(train_data, axis=0)[:max_samples]
        y = np.concatenate(train_labels, axis=0)[:max_samples]
        
        self.save_dataset('mnist', X, y, dataset_type='classification',
                         description='MNIST handwritten digits dataset')
        
        return X, y
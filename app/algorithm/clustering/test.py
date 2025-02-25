import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from app.algorithm.clustering.kmeans import KMeans, KMeansParams
from app.algorithm.clustering.dbscan import DBSCAN, DBSCANParams
from app.algorithm.clustering.hierarchical import Hierarchical, HierarchicalParams
from app.algorithm.clustering.gmm import GMM, GMMParams
from app.algorithm.clustering.train import evaluate_clustering

class TestClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 生成测试数据集
        n_samples = 300
        random_state = 42
        
        # 为每个算法生成合适的数据集
        # 1. K-means：球形分布数据
        kmeans_X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                                cluster_std=[0.5, 0.8, 0.3], random_state=random_state)
        
        # 2. DBSCAN：非球形分布数据
        dbscan_X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
        
        # 3. 层次聚类：同心圆数据
        hierarchical_X, _ = make_circles(n_samples=n_samples, noise=0.03, factor=0.5, random_state=random_state)
        
        # 4. GMM：混合高斯分布数据
        gmm_X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                             cluster_std=[0.5, 0.8, 1.0], random_state=random_state)
        
        # 划分训练集和测试集
        cls.datasets = {
            'KMeans': train_test_split(kmeans_X, test_size=0.2, random_state=random_state),
            'DBSCAN': train_test_split(dbscan_X, test_size=0.2, random_state=random_state),
            'Hierarchical': train_test_split(hierarchical_X, test_size=0.2, random_state=random_state),
            'GMM': train_test_split(gmm_X, test_size=0.2, random_state=random_state)
        }

    # 修改test_kmeans方法
    def test_kmeans(self):
        # 测试K-means聚类
        params = KMeansParams(n_clusters=3)
        model = KMeans(params)
        
        try:
            X_train, X_test = self.datasets['KMeans']
            # 训练模型
            history = model.fit(X_train)
            self.assertIsNotNone(history)
            self.assertIn('inertia', history)
            self.assertIn('centroid_shifts', history)
            
            # 预测
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)
            
            # 评估
            train_metrics = evaluate_clustering(X_train, train_labels, 'KMeans', history)
            test_metrics = evaluate_clustering(X_test, test_labels, 'KMeans', history)
            
            # 验证基础评估指标
            self.assertGreater(train_metrics['silhouette'], 0)
            self.assertGreater(train_metrics['calinski_harabasz'], 0)
            self.assertGreater(test_metrics['silhouette'], 0)
            self.assertGreater(test_metrics['calinski_harabasz'], 0)
            
            # 验证K-means特有指标
            self.assertIsNotNone(train_metrics.get('inertia'))
            self.assertIsInstance(train_metrics.get('silhouette_history'), list)
        finally:
            model.dispose()
    
    # 修改test_dbscan方法
    def test_dbscan(self):
        # 测试DBSCAN聚类
        params = DBSCANParams(eps=0.2, min_samples=5)  # 调整eps参数以适应月牙形数据集
        model = DBSCAN(params)
        
        try:
            X_train, X_test = self.datasets['DBSCAN']
            # 训练模型
            history = model.fit(X_train)
            self.assertIsNotNone(history)
            
            # 预测
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)
            
            # 评估
            train_metrics = evaluate_clustering(X_train, train_labels, 'DBSCAN', history)
            test_metrics = evaluate_clustering(X_test, test_labels, 'DBSCAN', history)
            
            # 验证基础评估指标
            self.assertGreaterEqual(train_metrics['silhouette'], 0)
            self.assertGreaterEqual(train_metrics['calinski_harabasz'], 0)
            self.assertGreaterEqual(test_metrics['silhouette'], 0)
            self.assertGreaterEqual(test_metrics['calinski_harabasz'], 0)
            
            # 验证DBSCAN特有指标
            self.assertIsInstance(train_metrics['noise_ratio'], float)
        finally:
            model.dispose()
    
    # 修改test_hierarchical方法
    def test_hierarchical(self):
        # 测试层次聚类
        params = HierarchicalParams(n_clusters=2, linkage='average')  # 使用average链接方法以更好地处理同心圆
        model = Hierarchical(params)
        
        try:
            X_train, X_test = self.datasets['Hierarchical']
            # 训练模型
            history = model.fit(X_train)
            self.assertIsNotNone(history)
            self.assertIn('n_clusters', history)
            self.assertIn('merged_clusters', history)
            
            # 预测
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)
            
            # 评估
            train_metrics = evaluate_clustering(X_train, train_labels, 'Hierarchical', history)
            test_metrics = evaluate_clustering(X_test, test_labels, 'Hierarchical', history)
            
            # 验证基础评估指标
            self.assertGreater(train_metrics['silhouette'], 0)
            self.assertGreater(train_metrics['calinski_harabasz'], 0)
            self.assertGreater(test_metrics['silhouette'], 0)
            self.assertGreater(test_metrics['calinski_harabasz'], 0)
            
            # 验证层次聚类特有指标
            self.assertIsInstance(train_metrics.get('cophenetic_correlation'), float)
        finally:
            model.dispose()
    
    # 修改test_gmm方法
    def test_gmm(self):
        # 测试高斯混合模型
        params = GMMParams(n_components=3, covariance_type='full')  # 使用完整协方差矩阵
        model = GMM(params)
        
        try:
            X_train, X_test = self.datasets['GMM']
            # 训练模型
            history = model.fit(X_train)
            self.assertIsNotNone(history)
            self.assertIn('log_likelihood', history)
            self.assertIn('n_iter', history)
            
            # 预测
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)
            
            # 评估
            train_metrics = evaluate_clustering(X_train, train_labels, 'GMM', history)
            test_metrics = evaluate_clustering(X_test, test_labels, 'GMM', history)
            
            # 验证基础评估指标
            self.assertGreaterEqual(train_metrics['silhouette'], 0)
            self.assertGreaterEqual(train_metrics['calinski_harabasz'], 0)
            self.assertGreaterEqual(test_metrics['silhouette'], 0)
            self.assertGreaterEqual(test_metrics['calinski_harabasz'], 0)
        finally:
            model.dispose()

if __name__ == '__main__':
    unittest.main()
import torch
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from app.algorithm.clustering.kmeans import KMeans, KMeansParams
from app.algorithm.clustering.dbscan import DBSCAN, DBSCANParams
from app.algorithm.clustering.hierarchical import Hierarchical, HierarchicalParams
from app.algorithm.clustering.gmm import GMM, GMMParams

def load_data():
    # 生成不同类型的数据集
    n_samples = 1000
    random_state = 42
    
    # 1. 球形分布数据集 (适用于K-means)
    kmeans_X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                            cluster_std=[0.5, 0.8, 0.3],  # 保持原有参数
                            random_state=random_state)
    
    # 2. 非球形分布数据集 (适用于DBSCAN)
    dbscan_X, _ = make_moons(n_samples=n_samples, noise=0.05,  # 降低噪声使结构更清晰
                            random_state=random_state)
    
    # 3. 层次结构数据集 (适用于层次聚类)
    hierarchical_X, _ = make_circles(n_samples=n_samples, noise=0.03,
                                   factor=0.5,  # 调整内外圆的比例
                                   random_state=random_state)
    
    # 4. 混合高斯分布数据集 (适用于GMM)
    gmm_X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3,
                         cluster_std=[0.5, 0.8, 1.0],  # 调整方差差异
                         random_state=random_state)
    
    # 划分训练集和测试集
    kmeans_train, kmeans_test = train_test_split(kmeans_X, test_size=0.2, random_state=random_state)
    dbscan_train, dbscan_test = train_test_split(dbscan_X, test_size=0.2, random_state=random_state)
    hierarchical_train, hierarchical_test = train_test_split(hierarchical_X, test_size=0.2, random_state=random_state)
    gmm_train, gmm_test = train_test_split(gmm_X, test_size=0.2, random_state=random_state)
    
    return {
        'KMeans': (kmeans_train, kmeans_test),
        'DBSCAN': (dbscan_train, dbscan_test),
        'Hierarchical': (hierarchical_train, hierarchical_test),
        'GMM': (gmm_train, gmm_test)
    }

def evaluate_clustering(X, labels, algorithm_name=None, history=None):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        base_metrics = {
            'silhouette': 0,
            'calinski_harabasz': 0
        }
        
        # 即使只有一个类别，也返回算法特定的指标
        if algorithm_name == 'DBSCAN':
            base_metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)
        elif algorithm_name == 'GMM' and history:
            if 'bic' in history:
                base_metrics['bic'] = history['bic'][-1] if history['bic'] else None
            if 'aic' in history:
                base_metrics['aic'] = history['aic'][-1] if history['aic'] else None
        
        return base_metrics
    
    # 基础评估指标
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels)
    }
    
    # 为DBSCAN添加噪声点比例指标和密度分布
    if algorithm_name == 'DBSCAN':
        noise_ratio = np.sum(labels == -1) / len(labels)
        metrics['noise_ratio'] = float(noise_ratio)
        if history and 'density_distribution' in history:
            metrics['density_distribution'] = history['density_distribution'][-1] if history['density_distribution'] else []
    
    # 其他算法特定的评估指标
    if history and algorithm_name:
        if algorithm_name == 'KMeans':
            # K-means特有指标：惯性值和轮廓系数历史
            if 'inertia' in history:
                metrics['inertia'] = history['inertia'][-1] if history['inertia'] else None
            if 'silhouette_scores' in history:
                metrics['silhouette_history'] = history['silhouette_scores']
        elif algorithm_name == 'Hierarchical':
            # 层次聚类特有指标：Cophenetic相关系数
            if 'cophenetic_correlation' in history:
                metrics['cophenetic_correlation'] = history['cophenetic_correlation']
        elif algorithm_name == 'GMM':
            # GMM特有指标：BIC和AIC
            if 'bic' in history:
                metrics['bic'] = history['bic']
            if 'aic' in history:
                metrics['aic'] = history['aic']
    
    return metrics

def train_and_evaluate():
    # 加载数据
    datasets = load_data()
    
    # 定义要测试的算法和参数，调整参数以获得更好的聚类效果
    algorithms = [
        (KMeans, KMeansParams(n_clusters=3, max_iter=500)),  # 保持原有参数
        (DBSCAN, DBSCANParams(eps=0.2, min_samples=5)),  # 减小eps值以适应更紧密的月牙形数据集
        (Hierarchical, HierarchicalParams(n_clusters=2, linkage='average')),  # 使用average链接方法以更好地处理同心圆
        (GMM, GMMParams(n_components=3, covariance_type='full', max_iter=200))  # 增加组件数量并使用full协方差类型
    ]
    
    results = {}
    
    for Algorithm, params in algorithms:
        algo_name = Algorithm.__name__
        # 获取对应算法的数据集
        X_train, X_test = datasets[algo_name]
        
        # 初始化模型
        model = Algorithm(params)
        
        # 训练模型
        print(f"\n训练 {algo_name}...")
        history = model.fit(X_train)
        
        # 在训练集上评估
        train_labels = model.predict(X_train)
        train_metrics = evaluate_clustering(X_train, train_labels, algo_name, history)
        
        # 在测试集上评估
        test_labels = model.predict(X_test)
        test_metrics = evaluate_clustering(X_test, test_labels, algo_name, history)
        
        # 保存结果
        results[algo_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history,
            'params': model.get_params()
        }
        
        # 清理内存
        model.dispose()
    
    return results

if __name__ == '__main__':
    # 运行训练和评估
    results = train_and_evaluate()
    
    # 打印结果
    for algo_name, result in results.items():
        print(f"\n{algo_name} 结果:")
        print("训练集评估:")
        print(f"  轮廓系数: {result['train_metrics']['silhouette']:.3f}")
        print(f"  Calinski-Harabasz指数: {result['train_metrics']['calinski_harabasz']:.3f}")
        
        # 打印算法特定的评估指标
        if algo_name == 'KMeans':
            print(f"  惯性值: {result['train_metrics'].get('inertia', 'N/A')}")
            print(f"  最终轮廓系数: {result['train_metrics'].get('silhouette_history', [])[-1] if result['train_metrics'].get('silhouette_history') else 'N/A'}")
        elif algo_name == 'DBSCAN':
            print(f"  密度分布: {result['train_metrics'].get('density_distribution', 'N/A')}")
            print(f"  噪声点比例: {result['train_metrics'].get('noise_ratio', 'N/A')}")
        elif algo_name == 'GMM':
            print(f"  BIC: {result['train_metrics'].get('bic', 'N/A')}")
            print(f"  AIC: {result['train_metrics'].get('aic', 'N/A')}")
        
        print("测试集评估:")
        print(f"  轮廓系数: {result['test_metrics']['silhouette']:.3f}")
        print(f"  Calinski-Harabasz指数: {result['test_metrics']['calinski_harabasz']:.3f}")
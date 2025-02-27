import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RegressionDataManager:
    @staticmethod
    def _load_from_csv(file_path):
        """从CSV文件加载数据集"""
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 分离特征和目标变量（假设最后一列是目标变量）
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 数据预处理
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_california_housing():
        """加载加利福尼亚房价数据集"""
        return RegressionDataManager._load_from_csv('data/regression/california_housing.csv')
    
    @staticmethod
    def load_linear_regression():
        """加载线性回归数据集"""
        return RegressionDataManager._load_from_csv('data/regression/linear_regression.csv')
    
    @staticmethod
    def load_nonlinear_regression():
        """加载非线性回归数据集"""
        return RegressionDataManager._load_from_csv('data/regression/nonlinear_regression.csv')
    
    @staticmethod
    def generate_synthetic_data(n_samples=1000, n_features=5):
        """生成合成数据集用于测试"""
        np.random.seed(42)
        
        # 生成特征矩阵
        X = np.random.randn(n_samples, n_features)
        
        # 生成目标变量（带有一些非线性关系和噪声）
        y = 3 * X[:, 0] + np.sin(4 * X[:, 1]) + X[:, 2]**2 - 0.5 * X[:, 3] + 0.1 * X[:, 4] + np.random.randn(n_samples) * 0.1
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return X_train, X_test, y_train, y_test, feature_names
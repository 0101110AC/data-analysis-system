import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetRegressionModel,
    RegressionParams
)

# 设置基本绘图样式
plt.style.use('seaborn-v0_8')  # 使用seaborn样式提升图表美观度

# 设置全局绘图参数
plt.rcParams['lines.linewidth'] = 2.5  # 增加线条宽度
plt.rcParams['lines.markersize'] = 8  # 增加标记点大小
plt.rcParams['font.size'] = 12  # 设置默认字体大小
plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 设置轴标签字体大小

def load_data():
    # 生成随机数据集用于测试
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # 生成特征矩阵
    X = np.random.randn(n_samples, n_features)
    
    # 生成目标变量（带有一些非线性关系和噪声）
    y = 3 * X[:, 0] + np.sin(4 * X[:, 1]) + X[:, 2]**2 - 0.5 * X[:, 3] + 0.1 * X[:, 4] + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def train_and_evaluate_models():
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # 定义模型参数
    models = {
        'Linear Regression': LinearRegressionModel({}),
        'Ridge Regression': RidgeRegressionModel({'alpha': 1.0}),
        'Lasso Regression': LassoRegressionModel({'alpha': 1.0}),
        'ElasticNet Regression': ElasticNetRegressionModel({'alpha': 1.0, 'l1_ratio': 0.5})
    }
    
    results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # 训练模型
        history = model.train(X_train, y_train)
        
        # 评估模型
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        results[name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history
        }
        
        print(f"{name} Training Set Performance:")
        print(f"MSE: {train_metrics['mse']:.4f}")
        print(f"RMSE: {train_metrics['rmse']:.4f}")
        print(f"R2: {train_metrics['r2']:.4f}")
        
        print(f"\n{name} Test Set Performance:")
        print(f"MSE: {test_metrics['mse']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"R2: {test_metrics['r2']:.4f}")
    
    return results

def plot_training_history(results):
    plt.figure(figsize=(15, 7))
    
    # 设置颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        plt.plot(history['loss'], label=name, linewidth=3.0, marker='o', 
                 markersize=8, color=colors[i], alpha=0.9)
    
    plt.title('Training Loss Curve', fontsize=16, pad=15)
    plt.xlabel('Training Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 绘制R2分数曲线
    plt.subplot(1, 2, 2)
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        plt.plot(history['r2'], label=name, linewidth=3.0, marker='o', 
                 markersize=8, color=colors[i], alpha=0.9)
    
    plt.title('Training R2 Score Curve', fontsize=16, pad=15)
    plt.xlabel('Training Epochs', fontsize=14)
    plt.ylabel('R2 Score', fontsize=14)
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(pad=2.0)
    plt.show()

def plot_model_comparison(results):
    model_names = list(results.keys())
    train_r2 = [result['train_metrics']['r2'] for result in results.values()]
    test_r2 = [result['test_metrics']['r2'] for result in results.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    
    # 设置颜色
    colors = ['#2ecc71', '#3498db']
    
    bars1 = plt.bar(x - width/2, train_r2, width, label='Training Set', color=colors[0], alpha=0.8)
    bars2 = plt.bar(x + width/2, test_r2, width, label='Test Set', color=colors[1], alpha=0.8)
    
    plt.xlabel('Model', fontsize=12, labelpad=10)
    plt.ylabel('R2 Score', fontsize=12, labelpad=10)
    plt.title('Model Performance Comparison', fontsize=14, pad=20)
    
    # 设置刻度标签
    plt.xticks(x, model_names, rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # 美化图例
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
              loc='upper right', bbox_to_anchor=(1, 1))
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 训练和评估模型
    results = train_and_evaluate_models()
    
    # 绘制训练历史
    plot_training_history(results)
    
    # 绘制模型比较
    plot_model_comparison(results)
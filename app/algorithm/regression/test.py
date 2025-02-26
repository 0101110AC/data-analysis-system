import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from .models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetRegressionModel,
    RegressionParams
)
from ...data.regression_data import RegressionDataManager

# 设置基本绘图样式
plt.style.use('seaborn-v0_8')  # 使用seaborn样式提升图表美观度

# 设置全局绘图参数
plt.rcParams['lines.linewidth'] = 2.5  # 增加线条宽度
plt.rcParams['lines.markersize'] = 8  # 增加标记点大小
plt.rcParams['font.size'] = 12  # 设置默认字体大小
plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 设置轴标签字体大小

def train_and_evaluate_models():
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names = RegressionDataManager.generate_synthetic_data()
    
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
    for (name, result), color in zip(results.items(), colors):
        history = result['history']
        plt.plot(history['loss'], label=name, color=color)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    
    # 绘制R2分数曲线
    plt.subplot(1, 2, 2)
    for (name, result), color in zip(results.items(), colors):
        history = result['history']
        plt.plot(history['r2'], label=name, color=color)
    plt.title('R² Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results):
    plt.figure(figsize=(12, 6))
    
    models = list(results.keys())
    train_rmse = [result['train_metrics']['rmse'] for result in results.values()]
    test_rmse = [result['test_metrics']['rmse'] for result in results.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_rmse, width, label='Training RMSE', color='#2ecc71')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE', color='#e74c3c')
    
    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 训练和评估模型
        results = train_and_evaluate_models()
        
        # 绘制训练历史
        plot_training_history(results)
        
        # 绘制模型比较图
        plot_model_comparison(results)
        
    except Exception as e:
        print(f"测试过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main()
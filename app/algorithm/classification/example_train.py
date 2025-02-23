import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .decision_tree import DecisionTree
import asyncio

def main():
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # 将问题转换为二分类：是否是Setosa品种

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 创建并训练模型
    model = DecisionTree({
        'maxDepth': 5,
        'minSamplesSplit': 2,
        'criterion': 'gini'
    })

    try:
        # 训练模型并获取训练历史
        history = model.train(X_train.tolist(), y_train.tolist())

        # 在测试集上评估模型
        metrics = model.evaluate(X_test.tolist(), y_test.tolist())

        print("训练完成！")
        print(f"测试集上的性能指标：")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    finally:
        # 清理资源
        model.dispose()

if __name__ == "__main__":
    main()
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.algorithm.classification.decision_tree import DecisionTree, DecisionTreeParams
from app.algorithm.classification.svm import SVM, SVMParams
from app.algorithm.classification.gan import GAN
from app.algorithm.classification.lstm import LSTM, LSTMParams
import asyncio

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    
    # 将数据转换为numpy数组格式，并限制数据量
    train_data = []
    train_labels = []
    max_samples = 5000  # 限制样本数量
    sample_count = 0
    
    for data, label in train_loader:
        if sample_count >= max_samples:
            break
        train_data.append(data.numpy().reshape(-1, 784))
        train_labels.append(label.numpy())
        sample_count += len(data)
    
    train_data = np.concatenate(train_data, axis=0)[:max_samples]
    train_labels = np.concatenate(train_labels, axis=0)[:max_samples]
    
    return train_data, train_labels

def preprocess_data(X):
    # 对数据进行归一化处理
    X = X / 255.0  # 将像素值缩放到[0,1]范围
    X = 2 * X - 1  # 将范围调整到[-1,1]
    # 对于LSTM，需要增加时间步维度
    X_lstm = X.reshape(X.shape[0], 28, 28)  # 将784维向量重塑为28x28的图像序列
    return X, X_lstm

def train_decision_tree(X_train, X_test, y_train, y_test):
    params = DecisionTreeParams(maxDepth=5, minSamplesSplit=2, criterion='gini')
    model = DecisionTree(params)

    try:
        print("\n开始训练决策树模型...")
        print(f"训练数据维度: {X_train.shape}")
        history = model.train(X_train.tolist(), y_train.tolist())
        metrics = model.evaluate(X_test.tolist(), y_test.tolist())
        print("决策树模型训练完成！")
        print(f"测试集性能指标：")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        return history, metrics
    finally:
        model.dispose()

def train_svm(X_train, X_test, y_train, y_test):
    params = SVMParams(
        kernel='rbf',
        gamma='scale',
        max_iter=50  # 进一步减少训练轮数
    )
    model = SVM(params)

    try:
        print("\n开始训练SVM模型...")
        history = model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        print("SVM模型训练完成！")
        print(f"测试集性能指标：")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        return history, metrics
    finally:
        model.dispose()

def train_gan(X_train, X_test, y_train, y_test):
    model = GAN({
        'latent_dim': 100,
        'hidden_dim': 256,
        'output_dim': 784,  # 修正输出维度为784，匹配MNIST数据维度
        'lr': 0.0002,
        'beta1': 0.5
    })

    try:
        print("\n开始训练GAN模型...")
        history = model.train(X_train, num_epochs=100, batch_size=128)  # 减少训练轮数，增加批处理大小
        print("GAN模型训练完成！")
        print(f"生成器最终损失: {history['g_losses'][-1]:.4f}")
        print(f"判别器最终损失: {history['d_losses'][-1]:.4f}")
        return history
    finally:
        model.dispose()

def train_lstm(X_train, X_test, y_train, y_test):
    params = LSTMParams(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.1
    )
    model = LSTM(params)

    try:
        print("\n开始训练LSTM模型...")
        history = model.train(X_train, y_train, num_epochs=50, batch_size=64)  # 减少训练轮数，增加批处理大小
        metrics = model.evaluate(X_test, y_test)
        print("LSTM模型训练完成！")
        print(f"测试集性能指标：")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        return history, metrics
    finally:
        model.dispose()

def main():
    # 加载MNIST数据集
    print("加载数据集...")
    X, y = load_mnist_data()
    y = (y >= 5).astype(int)  # 将问题转换为二分类：数字是否大于等于5

    print("数据预处理...")
    # 数据预处理
    X_standard, X_lstm = preprocess_data(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_standard, y, test_size=0.2, random_state=42
    )
    
    # 为LSTM准备数据
    X_train_lstm, X_test_lstm = train_test_split(
        X_lstm, test_size=0.2, random_state=42
    )[0:2]

    # 训练所有模型
    try:
        # 训练决策树
        dt_history, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
        
        # 训练SVM
        svm_history, svm_metrics = train_svm(X_train, X_test, y_train, y_test)
        
        # 训练GAN
        gan_history = train_gan(X_train, X_test, y_train, y_test)
        
        # 训练LSTM
        lstm_history, lstm_metrics = train_lstm(X_train_lstm, X_test_lstm, y_train, y_test)

        # 打印所有模型的比较结果
        print("\n模型性能比较：")
        print("决策树 - 准确率: {:.4f}".format(dt_metrics['accuracy']))
        print("SVM - 准确率: {:.4f}".format(svm_metrics['accuracy']))
        print("LSTM - 准确率: {:.4f}".format(lstm_metrics['accuracy']))

    except Exception as e:
        print(f"训练过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main()
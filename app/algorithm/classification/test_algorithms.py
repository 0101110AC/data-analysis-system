import torch
import numpy as np
from torchvision import datasets, transforms
from gan import GAN, GANParams
from svm import SVM, SVMParams
from lstm import LSTM, LSTMParams
from decision_tree import DecisionTree, DecisionTreeParams

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    
    # 将数据转换为numpy数组格式
    train_data = []
    train_labels = []
    for data, label in train_loader:
        train_data.append(data.numpy().reshape(-1, 784))
        train_labels.append(label.numpy())
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    return train_data, train_labels

def test_gan():
    print('\n测试GAN模型...')
    train_data, _ = load_mnist_data()
    
    # 初始化GAN模型
    gan_params = GANParams(latent_dim=100, hidden_dim=256, output_dim=784)
    gan = GAN(gan_params)
    
    # 训练模型
    history = gan.train(train_data[:1000], num_epochs=5, batch_size=64)
    
    # 生成样本
    generated_samples = gan.generate(10)
    
    print('GAN训练完成')
    print(f'生成器损失: {history["g_losses"][-1]:.4f}')
    print(f'判别器损失: {history["d_losses"][-1]:.4f}')
    
    gan.dispose()

def test_svm():
    print('\n测试SVM模型...')
    train_data, train_labels = load_mnist_data()
    
    # 只使用两个类别进行二分类测试
    mask = (train_labels == 0) | (train_labels == 1)
    X = train_data[mask][:1000]
    y = train_labels[mask][:1000]
    
    # 初始化SVM模型
    svm_params = SVMParams(kernel='rbf', C=1.0)
    svm = SVM(svm_params)
    
    # 训练模型
    history = svm.train(X, y)
    
    # 评估模型
    metrics = svm.evaluate(X[:100], y[:100])
    
    print('SVM训练完成')
    print(f'测试准确率: {metrics["accuracy"]:.4f}')
    print(f'测试损失: {metrics["loss"]:.4f}')
    
    svm.dispose()

def test_lstm():
    print('\n测试LSTM模型...')
    train_data, train_labels = load_mnist_data()
    
    # 重塑数据为序列格式 (batch_size, sequence_length, input_size)
    X = train_data[:1000].reshape(-1, 28, 28)  # 将每行像素视为序列
    y = (train_labels[:1000] >= 5).astype(np.float32)  # 二分类问题：是否大于等于5
    
    # 初始化LSTM模型
    lstm_params = LSTMParams(input_size=28, hidden_size=64, num_layers=2)
    lstm = LSTM(lstm_params)
    
    # 训练模型
    history = lstm.train(X, y, num_epochs=5)
    
    # 评估模型
    metrics = lstm.evaluate(X[:100], y[:100])
    
    print('LSTM训练完成')
    print(f'测试准确率: {metrics["accuracy"]:.4f}')
    print(f'测试损失: {metrics["loss"]:.4f}')
    
    lstm.dispose()

def test_decision_tree():
    print('\n测试决策树模型...')
    train_data, train_labels = load_mnist_data()
    
    # 只使用两个类别进行二分类测试
    mask = (train_labels == 0) | (train_labels == 1)
    X = train_data[mask][:1000]
    y = train_labels[mask][:1000]
    
    # 初始化决策树模型
    dt_params = DecisionTreeParams(maxDepth=5, minSamplesSplit=2)
    dt = DecisionTree(dt_params)
    
    # 训练模型
    history = dt.train(X, y)
    
    # 评估模型
    metrics = dt.evaluate(X[:100], y[:100])
    
    print('决策树训练完成')
    print(f'测试准确率: {metrics["accuracy"]:.4f}')
    print(f'测试损失: {metrics["loss"]:.4f}')
    
    dt.dispose()

def main():
    print('开始测试机器学习算法...')
    
    try:
        test_decision_tree()
        test_gan()
        test_svm()
        test_lstm()
        
        print('\n所有算法测试完成！')
    except Exception as e:
        print(f'测试过程中出现错误: {str(e)}')

if __name__ == '__main__':
    main()
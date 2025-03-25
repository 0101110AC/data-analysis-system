# 数据分析系统后端

## 项目简介

这是一个基于Python的机器学习数据分析系统后端，提供了丰富的机器学习算法实现和数据分析功能。系统采用现代化的技术栈，支持多种机器学习模型的训练、评估和预测。

## 系统架构

系统采用模块化设计，主要包含以下核心组件：

- **算法模块**：实现了多种机器学习算法
  - 分类算法：LSTM、决策树、SVM、GAN等
  - 聚类算法：K-means、DBSCAN、GMM、层次聚类
  - 回归算法：线性回归、非线性回归等

- **数据管理**：支持多种数据集的管理和预处理
  - 内置数据集：MNIST、加州房价等
  - 自定义数据集：支持CSV格式

- **AI代理**：集成了基于LangChain的ML专家系统
  - 支持本地模型和云端API
  - 提供专业的机器学习咨询和指导

## 主要特性

- **丰富的算法实现**
  - 深度学习：LSTM with Attention机制
  - 传统机器学习：决策树、SVM等
  - 无监督学习：多种聚类算法

- **灵活的部署选项**
  - 支持GPU加速（CUDA）
  - 可选择本地或云端AI模型

- **完善的API接口**
  - RESTful API设计
  - 支持流式响应
  - 异步处理能力

## 安装部署

### 环境要求

- Python 3.8+
- PyTorch
- FastAPI
- LangChain

### 安装步骤

1. 克隆项目
```bash
git clone [项目地址]
cd 数据分析系统后端
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 启动服务
```bash
python main.py
```

## 目录结构

```
├── algorithm/           # 算法实现
│   ├── classification/  # 分类算法
│   ├── clustering/      # 聚类算法
│   └── regression/      # 回归算法
├── data/               # 数据集
│   ├── MNIST/          # MNIST数据集
│   ├── clustering/      # 聚类数据
│   └── regression/      # 回归数据
├── lib/                # 工具库
│   ├── ml_agent.py     # AI代理实现
│   └── doubao_1_5_pro_32k.py
└── main.py            # 主程序入口
```

## API文档

### ML代理接口

```python
POST /api/ml-agent
Content-Type: application/json

{
    "messages": [
        {"role": "user", "content": "如何选择合适的机器学习算法？"}
    ]
}
```

### 模型训练接口

```python
POST /api/train
Content-Type: application/json

{
    "model_type": "lstm",
    "params": {
        "input_size": 784,
        "hidden_size": 128,
        "num_layers": 2
    },
    "data": {
        "x_train": [...],
        "y_train": [...]
    }
}
```

## 开发指南

### 添加新算法

1. 在相应目录下创建算法实现文件
2. 实现标准接口（train、predict、evaluate）
3. 在main.py中注册新算法

### 代码规范

- 遵循PEP 8编码规范
- 使用类型注解
- 编写单元测试

## 性能优化

- 使用PyTorch进行GPU加速
- 支持批处理和异步处理
- 内存管理优化

## 许可证

[选择合适的开源许可证]

## 贡献指南

欢迎提交Issue和Pull Request

## 联系方式

[添加联系方式]
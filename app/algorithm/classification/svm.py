import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SVMParams:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', max_iter=1000):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter

class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        return torch.mean(torch.clamp(1 - output * target, min=0))

class SVMModel(nn.Module):
    def __init__(self, input_dim, kernel='rbf', gamma='scale'):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.input_dim = input_dim
        
        # 定义网络结构
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),  # 使用input_dim作为输入维度
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        if self.kernel == 'linear':
            return self.layers(x)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (self.input_dim * x.var())
            else:
                gamma = self.gamma
            
            # 计算RBF核
            x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
            x_t = torch.transpose(x, 0, 1)
            dist = x_norm + torch.transpose(x_norm, 0, 1) - 2.0 * torch.mm(x, x_t)
            kernel_matrix = torch.exp(-gamma * dist)
            
            # 直接使用原始输入进行预测
            return self.layers(x)

class SVM:
    def __init__(self, params=None):
        if params is None:
            params = SVMParams()
        self.kernel = params.kernel
        self.C = params.C
        self.gamma = params.gamma
        self.max_iter = params.max_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
    def train(self, x, y):
        # 将数据转换为PyTorch张量，并确保维度正确
        x = x.reshape(-1, 784)  # 将输入数据reshape为(batch_size, 784)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # 初始化模型
        self.model = SVMModel(x.shape[1], self.kernel, self.gamma).to(self.device)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = HingeLoss()
        
        history = {
            'loss': [],
            'accuracy': []
        }
        
        batch_size = 32
        
        for epoch in range(self.max_iter):
            self.model.train()
            total_loss = 0
            correct_preds = 0
            
            # Mini-batch训练
            for i in range(0, len(x), batch_size):
                batch_x = x_tensor[i:min(i + batch_size, len(x))]
                batch_y = y_tensor[i:min(i + batch_size, len(x))]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, 2 * batch_y - 1)  # 将标签从[0,1]映射到[-1,1]
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_preds += ((outputs > 0) == batch_y).sum().item()
            
            epoch_loss = total_loss / (len(x) / batch_size)
            epoch_accuracy = correct_preds / len(x)
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
        
        del x_tensor
        del y_tensor
        
        return history
    
    def predict(self, x):
        self.model.eval()
        x = x.reshape(-1, 784)  # 将输入数据reshape为(batch_size, 784)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x_tensor)
            predictions = (outputs > 0).cpu().numpy()
        
        del x_tensor
        return predictions.reshape(-1)
    
    def evaluate(self, x_test, y_test):
        self.model.eval()
        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x_tensor)
            predictions = outputs > 0
            accuracy = (predictions == y_tensor).sum().item() / len(y_test)
            
            criterion = HingeLoss()
            loss = criterion(outputs, 2 * y_tensor - 1)
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
        del x_tensor
        del y_tensor
        del outputs
        
        return metrics
    
    def dispose(self):
        self.model = None
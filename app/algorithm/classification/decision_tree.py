import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class DecisionTreeParams:
    def __init__(self, maxDepth=None, minSamplesSplit=None, criterion=None):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.criterion = criterion

class TreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),  # 调整输入维度为784（MNIST数据集的特征维度）
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class DecisionTree:
  def __init__(self, params=None):
    if params is None:
      params = DecisionTreeParams()
    self.maxDepth = params.maxDepth if params.maxDepth is not None else 5
    self.minSamplesSplit = params.minSamplesSplit if params.minSamplesSplit is not None else 2
    self.criterion = params.criterion if params.criterion is not None else 'gini'
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    self.model = TreeModel()
    self.model.to(self.device)



  def train(self, x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)

    optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    history = {
      'loss': [],
      'accuracy': []
    }

    num_epochs = 100
    batch_size = 32

    for epoch in range(num_epochs):
      self.model.train()
      total_loss = 0
      correct_preds = 0

      # Implement mini-batch training
      for i in range(0, len(x), batch_size):
        batch_x = x_tensor[i:min(i + batch_size, len(x))]
        batch_y = y_tensor[i:min(i + batch_size, len(x))]

        optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_preds += ((outputs > 0.5) == batch_y).sum().item()

      epoch_loss = total_loss / math.ceil(len(x) / batch_size)
      epoch_accuracy = correct_preds / len(x)

      history['loss'].append(epoch_loss)
      history['accuracy'].append(epoch_accuracy)

    del x_tensor
    del y_tensor

    return history

  def predict(self, x):
    self.model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
    
    predictions = self.model(x_tensor)
    results = predictions.cpu().detach().numpy()
    
    del x_tensor
    del predictions

    return [r[0] for r in results]

  def evaluate(self, x_test, y_test):
    self.model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)

    criterion = nn.BCELoss()
    
    outputs = self.model(x_tensor)
    loss = criterion(outputs, y_tensor)
    predictions = outputs > 0.5
    accuracy = (predictions == y_tensor).sum().item() / len(y_test)

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

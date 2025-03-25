import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from abc import ABC, abstractmethod

class BaseRegression(ABC):
    def __init__(self, params):
        self.model = None
        self.params = params
        self.history = {'loss': [], 'r2': []}
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2)
        }
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.params
    
    def dispose(self):
        self.model = None
        self.history = None

class LinearRegressionModel(BaseRegression):
    def __init__(self, params):
        super().__init__(params)
        self.model = LinearRegression(**{k: v for k, v in params.items()})
    
    def train(self, X, y):
        # 使用小批量梯度下降进行迭代训练
        n_samples = len(X)
        batch_size = min(200, n_samples)
        n_epochs = 10
        
        for epoch in range(n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 批量训练
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:min(i + batch_size, n_samples)]
                batch_y = y_shuffled[i:min(i + batch_size, n_samples)]
                self.model.fit(batch_X, batch_y)
            
            # 记录每个epoch的性能指标
            metrics = self.evaluate(X, y)
            self.history['loss'].append(metrics['mse'])
            self.history['r2'].append(metrics['r2'])
        
        return self.history

class RidgeRegressionModel(BaseRegression):
    def __init__(self, params):
        super().__init__(params)
        self.model = Ridge(**{k: v for k, v in params.items()})
    
    def train(self, X, y):
        # 使用小批量梯度下降进行迭代训练
        n_samples = len(X)
        batch_size = min(200, n_samples)
        n_epochs = 10
        
        for epoch in range(n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 批量训练
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:min(i + batch_size, n_samples)]
                batch_y = y_shuffled[i:min(i + batch_size, n_samples)]
                self.model.fit(batch_X, batch_y)
            
            # 记录每个epoch的性能指标
            metrics = self.evaluate(X, y)
            self.history['loss'].append(metrics['mse'])
            self.history['r2'].append(metrics['r2'])
        
        return self.history

class LassoRegressionModel(BaseRegression):
    def __init__(self, params):
        super().__init__(params)
        self.model = Lasso(**{k: v for k, v in params.items()})
    
    def train(self, X, y):
        # 使用小批量梯度下降进行迭代训练
        n_samples = len(X)
        batch_size = min(200, n_samples)
        n_epochs = 10
        
        for epoch in range(n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 批量训练
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:min(i + batch_size, n_samples)]
                batch_y = y_shuffled[i:min(i + batch_size, n_samples)]
                self.model.fit(batch_X, batch_y)
            
            # 记录每个epoch的性能指标
            metrics = self.evaluate(X, y)
            self.history['loss'].append(metrics['mse'])
            self.history['r2'].append(metrics['r2'])
        
        return self.history

class ElasticNetRegressionModel(BaseRegression):
    def __init__(self, params):
        super().__init__(params)
        self.model = ElasticNet(**{k: v for k, v in params.items()})
    
    def train(self, X, y):
        # 使用小批量梯度下降进行迭代训练
        n_samples = len(X)
        batch_size = min(200, n_samples)
        n_epochs = 10
        
        for epoch in range(n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 批量训练
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:min(i + batch_size, n_samples)]
                batch_y = y_shuffled[i:min(i + batch_size, n_samples)]
                self.model.fit(batch_X, batch_y)
            
            # 记录每个epoch的性能指标
            metrics = self.evaluate(X, y)
            self.history['loss'].append(metrics['mse'])
            self.history['r2'].append(metrics['r2'])
        
        return self.history

class RegressionParams:
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def __iter__(self):
        return iter(self.params.items())
    
    def __getitem__(self, key):
        return self.params[key]
    
    def items(self):
        return self.params.items()
    
    def get(self, key, default=None):
        return self.params.get(key, default)
    
    def __len__(self):
        return len(self.params)
    
    def __bool__(self):
        return bool(self.params)
    
    def __getattr__(self, name):
        return self.params.get(name)
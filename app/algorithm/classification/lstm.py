import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMParams:
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1, attention=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention = attention

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # 注意力加权求和
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        if attention:
            self.attention_layer = AttentionLayer(hidden_size)
            
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
            
        lstm_out, (hidden_state, cell_state) = self.lstm(x, hidden)
        
        if self.attention:
            context, attention_weights = self.attention_layer(lstm_out)
            output = self.fc(context)
        else:
            output = self.fc(lstm_out[:, -1, :])
            attention_weights = None
            
        return self.sigmoid(output), attention_weights, (hidden_state, cell_state)

class LSTM:
    def __init__(self, params=None):
        if params is None:
            params = LSTMParams()
        self.input_size = params.input_size
        self.hidden_size = params.hidden_size
        self.num_layers = params.num_layers
        self.dropout = params.dropout
        self.attention = params.attention
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = LSTMModel(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.attention
        ).to(self.device)
        
    def train(self, x, y, num_epochs=100, batch_size=32, learning_rate=0.001):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        history = {
            'loss': [],
            'accuracy': [],
            'attention_weights': []
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct_preds = 0
            
            for i in range(0, len(x), batch_size):
                batch_x = x_tensor[i:min(i + batch_size, len(x))]
                batch_y = y_tensor[i:min(i + batch_size, len(x))]
                
                optimizer.zero_grad()
                outputs, attention_weights, _ = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_preds += ((outputs > 0.5) == batch_y).sum().item()
                
                if attention_weights is not None:
                    history['attention_weights'].append(attention_weights.detach().cpu().numpy())
            
            epoch_loss = total_loss / (len(x) / batch_size)
            epoch_accuracy = correct_preds / len(x)
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
        
        del x_tensor
        del y_tensor
        
        return history
    
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs, attention_weights, _ = self.model(x_tensor)
            predictions = (outputs > 0.5).cpu().numpy()
        
        del x_tensor
        return predictions.reshape(-1), attention_weights
    
    def evaluate(self, x_test, y_test):
        self.model.eval()
        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            outputs, attention_weights, _ = self.model(x_tensor)
            predictions = outputs > 0.5
            accuracy = (predictions == y_tensor).sum().item() / len(y_test)
            
            criterion = nn.BCELoss()
            loss = criterion(outputs, y_tensor)
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None
        }
        
        del x_tensor
        del y_tensor
        
        return metrics
    
    def dispose(self):
        self.model = None
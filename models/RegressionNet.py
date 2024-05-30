import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def evaluate(self, data_loader):
        self.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        return avg_loss
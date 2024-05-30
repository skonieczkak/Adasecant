import torch
import torch.nn as nn

class CustomRegNet(nn.Module):
    def __init__(self, input_dim):
        super(CustomRegNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
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
import torch
import torch.nn as nn


class SimpleClassNet(nn.Module):
    def __init__(self):
        super(SimpleClassNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 4 input features, 10 neurons
        self.fc2 = nn.Linear(10, 3)  # 10 neurons, 3 output classes
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def evaluate(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
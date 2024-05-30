import torch
import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_units):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_units = num_units
        self.lin = nn.Linear(in_features, out_features * num_units)

    def forward(self, x):
        shape = x.size(0), self.out_features, self.num_units
        x = self.lin(x)
        maxout, _ = x.view(*shape).max(-1)
        return maxout
    
class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_units):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_units = num_units
        self.lin = nn.Linear(in_features, out_features * num_units)

    def forward(self, x):
        shape = x.size(0), self.out_features, self.num_units
        x = self.lin(x)
        maxout, _ = x.view(*shape).max(-1)
        return maxout

class MaxoutNet(nn.Module):
    def __init__(self):
        super(MaxoutNet, self).__init__()
        self.fc1 = Maxout(28*28, 256, 4)
        self.fc2 = Maxout(256, 128, 4)
        self.fc3 = Maxout(128, 64, 4)
        self.fc4 = Maxout(64, 32, 4)
        self.fc5 = Maxout(32, 16, 4)
        self.fc6 = nn.Linear(16, 10)  # Final output layer with no maxout
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

    def evaluate(self, data_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
            m.bias.data.fill_(0.01)
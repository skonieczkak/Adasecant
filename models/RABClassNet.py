import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        x += residual
        return F.relu(x)
    
class RABClassNet(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate=0.2):
        super(RABClassNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.res_block1 = ResidualAttentionBlock(64, 64)
        self.res_block2 = ResidualAttentionBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_shape[1] // 16) * (input_shape[2] // 16), 128)
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.drop4 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = F.relu(self.fc2(x))
        x = self.drop4(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

    def evaluate(self, data_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy

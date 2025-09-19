import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, input_size=784, hidden=128, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(1 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        
    def forward(self, x):
        # Apply a simple convolutional layer
        x = self.conv_layers(x)
        # Flatten the image
        x = x.view(x.size(0), -1)
        
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc2(x)
        return x
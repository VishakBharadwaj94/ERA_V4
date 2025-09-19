import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, input_size=784, hidden=512, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(32 * 13 * 13, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Apply a simple convolutional layer
        x = self.conv_layers(x)
        # Flatten the image
        x = x.view(x.size(0), -1)
        
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc2(x)
        return x
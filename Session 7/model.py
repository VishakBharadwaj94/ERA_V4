"""
model.py - CIFAR-10 CNN 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Net(nn.Module):
    """
    CIFAR-10 CNN
    - No MaxPooling (using strided and dilated convolutions)
    - Depthwise Separable Convolution in C2
    - Dilated Convolutions in C3
    - GAP + 1x1 Conv for classification
    - Total RF: 55 pixels
    - Parameters: ~199k
    """
    def __init__(self, dropout_value=0.05):
        super(Net, self).__init__()
        
        # C1: Input Block (32x32x3 -> 32x32x22)
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 14, kernel_size=3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(14, 22, kernel_size=3, padding=1, bias=False),  # RF: 5
            nn.BatchNorm2d(22),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        # C2: Depthwise Separable Block (32x32x22 -> 16x16x44)
        self.convblock2 = nn.Sequential(
            DepthwiseSeparableConv(22, 32, kernel_size=3, padding=1, bias=False),  # RF: 7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(32, 44, kernel_size=3, padding=1, bias=False),  # RF: 9
            nn.BatchNorm2d(44),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            # Strided convolution instead of MaxPool
            nn.Conv2d(44, 44, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 11, size: 16x16
            nn.BatchNorm2d(44),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        # C3: Dilated Convolution Block (16x16x44 -> 8x8x60)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(44, 52, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 15
            nn.BatchNorm2d(52),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(52, 52, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 23
            nn.BatchNorm2d(52),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(52, 60, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 31
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            # Strided convolution
            nn.Conv2d(60, 60, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 39, size: 8x8
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        # C40: Final convolution block (8x8x60 -> 8x8x56)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(60, 56, kernel_size=3, padding=1, bias=False),  # RF: 47
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),  # RF: 55
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(56, 10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
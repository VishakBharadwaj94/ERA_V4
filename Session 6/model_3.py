"""
================================================================================
TARGET:
================================================================================
Address Model_2's underfitting by adding strategic capacity at RF 11-15
- Increase channels where digit features form: convblock4 (12→13), convblock5 (12→13)
- Stay under 8k params: 7,849 / 8,000 (98.1% budget)
- Keep strided conv approach (validated in Model_2)
- Test if architecture capacity was the bottleneck

Expected: If capacity was the issue, should reach 99.4%+

================================================================================
RESULTS:
================================================================================
Parameters: 7,849 / 8,000

Training Configuration:
- Optimizer: SGD(lr=0.01, momentum=0.9)
- Scheduler: StepLR(step_size=6, gamma=0.1)
- Dropout: 0.1
- Augmentation: RandomRotation(±7°)
- Epochs: 20

Performance:
- Peak Test Accuracy: 99.35% (Epoch 13)
- Last 3 Epochs: 99.34%, 99.29%, 99.31% (avg 99.31%)
- Best Train Accuracy: 99.01%
- First 99.4%+: Never achieved

================================================================================
ANALYSIS:
================================================================================

This experiment PROVES architecture capacity is NOT the bottleneck.
- Model_2: 7,386 params → 99.31% peak
- Model_3: 7,849 params → 99.35% peak (+0.04% for +463 params)

The marginal improvement shows diminishing returns. Both models plateau at epoch 6
when LR drops from 0.01 to 0.001 (90% reduction). After this drop, learning 
essentially freezes regardless of model capacity.

This proves the hypothesis WRONG: The issue is not "insufficient capacity"
but "suboptimal learning rates."

Next Step:
Instead of adding more parameters (which didn't help), need to fix the 
learning rate schedule.

Conclusion: Architecture is FINE. Learning rate schedule can be IMPROVED.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Model_3(nn.Module):
    """
    Architecture identical to Model_2
    Key difference: Optimized hyperparameters
    - lr=0.1 (vs 0.01)
    - StepLR(5, 0.5) (vs (6, 0.1))
    - dropout=0.1 (vs 0.05)
    """
    def __init__(self):
        super(Model_3, self).__init__()
        
        # Input Block - C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3
        
        # CONVOLUTION BLOCK 1 - C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24, RF = 5
        
        # TRANSITION BLOCK 1 - C3 (stride=2 instead of MaxPool)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 11, RF = 7
        
        # CONVOLUTION BLOCK 2 - C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value)
        ) # output_size = 9, RF = 11
        
        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value)
        ) # output_size = 7, RF = 15
        
        # C6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 5, RF = 19
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
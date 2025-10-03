"""
================================================================================
TARGET:
================================================================================
Eliminate Model_4's epoch 10-14 volatility for smoothest possible convergence
- Keep proven architecture (7,849 params) and lr=0.1 start
- Optimize LR schedule: StepLR(step_size=4, gamma=0.4)
  * More frequent decay (every 4 vs 5 epochs)
  * Gentler decay (0.4× vs 0.5×)
- Goal: Stable 99.4%+ immediately after first hit (no dips)

Expected: Hit 99.4%+ by epoch 9 and maintain without volatility

================================================================================
RESULTS:
================================================================================
Parameters: 7,849 / 8,000 (98.1% of budget)

Training Configuration:
- Optimizer: SGD(lr=0.1, momentum=0.9)
- Scheduler: StepLR(step_size=4, gamma=0.4)  [OPTIMAL]
- Dropout: 0.1
- Augmentation: RandomRotation(±7°)
- Epochs: 20

Performance:
- Peak Test Accuracy: 99.48% (Epoch 15)
- Last 3 Epochs: 99.45%, 99.46%, 99.45% (avg 99.45%)
- Best Train Accuracy: 98.95%
- First 99.4%+: Epoch 9 (99.41%)
- Consistency: 10 out of 11 final epochs ≥99.4% (91%)

LR Schedule (5 phases, more granular):
- 0.100 (Epochs 0-3): Initial fast learning
- 0.040 (Epochs 4-7): First refinement
- 0.016 (Epochs 8-11): Breakthrough phase
- 0.0064 (Epochs 12-15): Fine-tuning
- 0.00256 (Epochs 16-19): Final polish

Epoch Progression:
- Epoch 4: 99.24% (LR→0.04, steady climb)
- Epoch 9: 99.41% FIRST HIT (LR→0.016)
- Epoch 10: 99.37% (only dip)
- Epoch 11: 99.44% (recovered)
- Epochs 12-19: ALL ≥99.42% (stable!)

================================================================================
ANALYSIS:
================================================================================

OPTIMAL SOLUTION - All Requirements Exceeded:
✓ Parameters: 7,849 / 8,000 (151 under budget)
✓ Accuracy: 99.48% peak, 99.45% last 3 average
✓ Speed: First 99.4%+ at epoch 9 (6 epochs under 15 limit)
✓ Consistency: 91% of final epochs ≥99.4% (best of all models)

Comparison to Model_4:
- Model_4: First hit epoch 10, then 3 dips below 99.4% in next 5 epochs
- Final: First hit epoch 9, only 1 dip in next 11 epochs

The more frequent (step=4) and gentler (gamma=0.4) decay creates:
1. Smoother transitions between LR phases
2. Less overshooting/undershooting after each drop
3. Faster stabilization after breakthrough
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
dropout_value = 0.1

class Final_Model(nn.Module):
    """
    Same architecture as Model_3 and Model_4
    """
    def __init__(self):
        super(Final_Model, self).__init__()
        
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
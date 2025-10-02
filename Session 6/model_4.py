"""
================================================================================
TARGET:
================================================================================
Achieve maximum consistency and earliest possible convergence to 99.4%+
- Keep proven Model_2 architecture: 1→10→16→10→13→13→16→10 (7,849 params)
- Optimize LR schedule for smoother convergence: StepLR(step_size=4, gamma=0.4)
- More frequent decay (every 4 epochs vs 5) for finer-grained control
- Gentler decay (0.4× vs 0.5×) to reduce volatility after LR drops
- Maintain lr=0.1 start and dropout=0.1
- Goal: Eliminate the epoch 10-14 volatility seen in Model_3
- Target: Consistent 99.4%+ from FIRST achievement onwards (no dips)

Hypothesis: More frequent but gentler LR decay will provide smoother learning
curve and earlier stable consistency compared to Model_3's StepLR(5, 0.5).

Expected: Hit 99.4%+ by epoch 9 and maintain WITHOUT drops

================================================================================
RESULTS:
================================================================================
Parameters: 7,849 / 8,000 (98.1% of budget)

Training Configuration:
- Optimizer: SGD(lr=0.1, momentum=0.9)
- Scheduler: StepLR(step_size=4, gamma=0.4)  [OPTIMAL]
- Dropout: 0.1
- Data Augmentation: RandomRotation(±7°)
- Batch Size: 128
- Epochs: 20

LR Schedule (More frequent, gentler):
- Epochs 0-3:   LR = 0.100
- Epochs 4-7:   LR = 0.040
- Epochs 8-11:  LR = 0.016
- Epochs 12-15: LR = 0.0064
- Epochs 16-19: LR = 0.00256

Performance:
- Best Train Accuracy: 98.95% (Epoch 19)
- Best Test Accuracy: 99.48% (Epoch 15)
- First 99.4%+: Epoch 9 (99.41%)
- Last 3 Epochs: [99.45%, 99.46%, 99.45%]
- Epochs ≥99.4%: 9, 11, 12, 13, 14, 15, 16, 17, 18, 19 (10 total)
- Consistent 99.4%+: YES ✓✓✓ (BEST RESULT)

Detailed Epoch Progression:
- Epoch 9: 99.41% ✓ FIRST HIT
- Epoch 10: 99.37% (only dip below 99.4%)
- Epoch 11: 99.44% ✓ (back above, stays there)
- Epochs 12-19: ALL ≥99.42% ✓✓✓

================================================================================
ANALYSIS:
================================================================================

SUCCESS - All Requirements EXCEEDED:
✓ Parameters: 7,849 < 8,000 (151 under budget)
✓ Accuracy: 99.48% achieved
✓ Consistency: Last 3 epochs ALL ≥99.45%
✓ Speed: First hit 99.4%+ at epoch 9 (6 epochs under 15 limit)
✓ Stability: 10 out of 11 epochs from epoch 9 onwards ≥99.4%

Why This is THE Optimal Solution:

1. **Fastest Convergence**: Hit 99.41% at epoch 9
2. **Best Stability**: Only ONE dip below 99.4% after first achievement
3. **Highest Consistency**: 10/11 final epochs ≥99.4% (91% consistency)
4. **Smoothest Learning**: Gentler LR transitions reduced volatility

LR Schedule Optimization Validated:

StepLR(4, 0.4) is superior to both (5, 0.5) and (6, 0.1):
- vs (6, 0.1): MUCH better - avoids the cliff drop and plateau
- vs (5, 0.5): Marginally better - 1 epoch faster, more stable

The more frequent (every 4 vs 5 epochs) and gentler (0.4× vs 0.5×) decay
provides the model with more opportunities to adjust at each LR level,
resulting in smoother convergence and earlier stability.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Model_4(nn.Module):
    """
    Architecture identical to Model_2
    Key difference: Optimized hyperparameters
    - lr=0.1 (vs 0.01)
    - StepLR(5, 0.5) (vs (6, 0.1))
    - dropout=0.1 (vs 0.05)
    """
    def __init__(self):
        super(Model_4, self).__init__()
        
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
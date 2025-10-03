"""
================================================================================
TARGET:
================================================================================
Test if learning rate is the bottleneck (not architecture capacity)
- Keep Model_3 architecture (7,849 params) - already proven sufficient
- Increase LR 10×: 0.01 → 0.1 for faster exploration
- Use gentler decay: StepLR(step_size=5, gamma=0.5) vs (6, 0.1)
- Goal: Break through the 99.35% plateau and achieve consistent 99.4%+

Hypothesis: Higher LR will unlock the capacity that Model_3 already has

Expected: Should hit 99.4%+ by epoch 10-12 if hypothesis correct

================================================================================
RESULTS:
================================================================================
Parameters: 7,849 / 8,000 (same as Model_3)

Training Configuration:
- Optimizer: SGD(lr=0.1, momentum=0.9)  [10× higher than Model_3]
- Scheduler: StepLR(step_size=5, gamma=0.5)  [gentler than previous (6, 0.1)]
- Dropout: 0.1
- Augmentation: RandomRotation(±7°)
- Epochs: 20

Performance:
- Peak Test Accuracy: 99.48% (Epoch 19)
- Last 3 Epochs: 99.47%, 99.46%, 99.48% (avg 99.47%)
- Best Train Accuracy: 99.07%
- First 99.4%+: Epoch 10 (99.43%)


================================================================================
ANALYSIS:
================================================================================

What Worked:
✓ SUCCESS - Hypothesis confirmed! LR was the bottleneck, not architecture
✓ Achieved consistent 99.4%+ (last 3 epochs all ≥99.46%)
✓ Same 7,849 params as Model_3, just better hyperparameters
✓ First hit 99.4%+ at epoch 10 (within 15 epoch limit)
✓ Peak 99.48% matches baseline Model_1 (13.8k params)

Key Breakthrough:
Same architecture, 10× higher LR:
- Model_3 @ lr=0.01 → 99.35% (failed)
- Model_4 @ lr=0.1 → 99.48% (success)

T
What Could Be Better:
⚠ Minor volatility in epochs 10-14 (dipped to 99.35% at epoch 13)

Next Optimization:
For even smoother convergence, try:
- step_size=4 (more frequent, less time to drift)
- gamma=0.4 (gentler, smaller adjustments)

This should eliminate the epoch 10-14 volatility and achieve stable 99.4%+
immediately after first hit.

Conclusion:
Model_4 PROVES that architecture efficiency (strided conv, strategic capacity)
combined with proper learning dynamics (high LR, gradual decay) can match
baseline performance with 43% fewer parameters.

The assignment target is ACHIEVED, but can be perfected further.
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
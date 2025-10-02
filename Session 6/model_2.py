"""
================================================================================
TARGET:
================================================================================
Reduce parameters to <8k by using strided convolutions instead of MaxPooling
- Replace MaxPool with stride=2 conv to learn optimal downsampling
- Reduce channel counts strategically: 1→10→16→10→12→12→16→10
- Maintain proper receptive field coverage for MNIST digit recognition
- Validate if strided convolution approach is parameter-efficient
- Test if reduced capacity (7.4k vs 13.8k params) can still learn well

Expected: Should reach ~99.3-99.4% if architecture is sound

================================================================================
RESULTS:
================================================================================
Parameters: 7,386 / 8,000 (92.3% of budget used)

Training Configuration:
- Optimizer: SGD(lr=0.01, momentum=0.9)
- Scheduler: StepLR(step_size=6, gamma=0.1)
- Dropout: 0.05
- Data Augmentation: RandomRotation(±7°)
- Batch Size: 128
- Epochs: 20

Performance:
- Best Train Accuracy: 98.94%
- Best Test Accuracy: 99.31% (Epoch 16)
- Last 3 Epochs: [99.25%, 99.29%, 99.29%]
- Consistent 99.4%+: NO ✗

Epoch-by-Epoch:
- Epoch 0: Train 88.08%, Test 96.92%
- Epoch 5: Train 98.50%, Test 99.12%
- Epoch 6: Train 98.78%, Test 99.25% (LR drop happens here)
- Epoch 10: Train 98.87%, Test 99.29%
- Epoch 16: Train 98.91%, Test 99.31% (peak)
- Epoch 19: Train 98.94%, Test 99.29%

================================================================================
ANALYSIS:
================================================================================

What Worked:
✓ Strided convolution approach is VALID - model learns effectively
✓ Successfully reduced parameters by 46.5% (13.8k → 7.4k)
✓ Model is very stable - no overfitting (train 98.94% < test 99.31%)
✓ Architecture depth is appropriate for MNIST complexity
✓ Stayed comfortably under 8k parameter budget (614 params to spare)
✓ GAP effectively replaces fully connected layers
✓ Receptive field progression (3→5→7→11→15→19→27) covers full digit

What Didn't Work:
✗ Failed to reach 99.4% target - peaked at 99.31% (0.09% short)
✗ Not consistent in last 3 epochs - fluctuates between 99.25-99.29%
✗ Model is UNDERFITTING - train (98.94%) significantly below test (99.31%)
✗ Plateau at epoch 6 after LR drop - never recovers momentum
✗ 20 epochs to reach peak, but still didn't achieve target

Key Insights:

1. **Underfitting, Not Capacity Problem**:
   The 0.37% gap (train 98.94% < test 99.31%) clearly indicates underfitting.
   The model hasn't learned as much as it could from the training data.
   This suggests either: (a) insufficient capacity, OR (b) suboptimal learning.

2. **Learning Rate Too Conservative**:
   Starting at lr=0.01 is very cautious. The model takes 6 epochs just to 
   reach 99.25%. After the LR drops to 0.001 at epoch 6, improvement stalls.
   The aggressive 10× reduction (gamma=0.1) essentially freezes learning.


3. **Strided Convolution Validates**:
   The approach of using stride=2 instead of MaxPooling works excellently.
   It's parameter-efficient and learns adaptive downsampling. This innovation
   should be kept in future iterations.

4. **Dropout 0.05 + Augmentation = Strong Regularization**:
   Combining dropout 0.05 with RandomRotation(±7°) creates significant
   regularization. For an underfitting model, this might be excessive.
   Standard dropout is 0.1, so 0.05 is lighter, but combined with rotation,
   it still makes training harder.


5. **Architecture is Sound**:
   The 6 conv layers + GAP + 1×1 output structure is efficient. The channel
   progression (1→10→16→10→12→12→16→10) makes sense:
   - Narrow at start (simple edges)
   - Bottleneck after downsampling (10 channels)
   - Gradual increase as RF grows
   - Wider before GAP for rich features



Proposed Solutions for Model_2:

- Increase lr to 0.05 or 0.1 (5-10× higher)
- Use gentler LR decay: gamma=0.5 or 0.4 (not 0.1)
- More frequent decay: step_size=4 or 5 (not 6)

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.05  # Note: This was 0.1 in Model_1, reduced to 0.05 here so as to not weaken model too much

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
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
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 9, RF = 11
        
        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 7, RF = 15
        
        # C6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
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
        x = self.convblock3(x)  # Strided conv instead of MaxPool
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
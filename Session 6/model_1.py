"""
Original Model from code_10.py
Parameters: 13,808
This was the baseline before optimization

================================================================================
TARGET:
================================================================================
- Establish baseline performance with sufficient model capacity
- Achieve 99.4%+ test accuracy consistently
- Use standard architecture: MaxPooling + multiple conv layers + GAP
- Validate that the training pipeline works correctly
- Set the benchmark that optimized models need to match with fewer parameters

================================================================================
RESULTS:
================================================================================
Parameters: 13,808
Best Train Accuracy: 99.21% (Epoch 20)
Best Test Accuracy: 99.48% (Epoch 20)
Epochs when hit 99.4%+: 9th (99.45%), 20th (99.48%)
Last 3 Epochs Test Accuracy: [99.47%, 99.46%, 99.48%]
Consistent 99.4%+ in last 3 epochs: YES ✓

Training Configuration:
- Optimizer: SGD(lr=0.01, momentum=0.9)
- Scheduler: StepLR(step_size=6, gamma=0.1)
- Dropout: 0.1
- Data Augmentation: RandomRotation(±7°)
- Batch Size: 128
- Epochs: 20

Architecture:
- Input Block: 1→16 channels (RF=3)
- Conv Block 1: 16→32 channels (RF=5)
- Transition: 32→10 with 1x1 conv + MaxPool(2,2) (RF=10)
- Conv Block 2: 10→16→16→16→16 channels (RF=12,14,16,16)
- Output: GAP + 1x1 conv to 10 classes

================================================================================
ANALYSIS:
================================================================================

What Worked:
✓ Successfully achieved consistent 99.4%+ accuracy
✓ Model has sufficient capacity (13.8k parameters)
✓ Training pipeline is solid (dropout, BatchNorm, GAP all working)
✓ Data augmentation (rotation) helps generalization
✓ GAP instead of fully connected layers prevents parameter explosion
✓ 1x1 transition conv effectively reduces channels before pooling

What Could Be Improved:
✗ Parameter count (13.8k) is 72% over the <8k target
✗ Uses MaxPooling (fixed operation, no learning)
✗ Conservative learning rate (0.01) takes 20 epochs to converge
✗ Has 4 layers after pooling - may be redundant
✗ Channel distribution not optimized (16→32 early, then 16 throughout)

Key Insights:
1. The model has MORE capacity than needed - it achieves 99.21% train accuracy
   but 99.48% test accuracy, indicating potential overfitting that's controlled
   by dropout and augmentation.

2. MaxPooling at layer 3 is a fixed operation. Modern architectures use strided
   convolutions to LEARN the downsampling, which could be more parameter-efficient.

3. The 16 channels in the middle layers (convblock4-7) after the 10-channel
   transition might be excessive. The receptive field at these layers can see
   most of the digit, so fewer channels might suffice.

4. Four sequential conv layers (convblock4-7) after pooling seems redundant.
   Could potentially reduce to 3 layers and still capture the necessary features.

5. Learning rate of 0.01 is very conservative. Takes 9 epochs to first hit 99.4%.
   A higher learning rate might converge faster.

Next Steps for Optimization:
→ Replace MaxPooling with strided convolution (learnable downsampling)
→ Reduce channel counts strategically: keep low where RF is small, increase
  where RF sees complete digits
→ Remove one conv layer to reduce parameters
→ Experiment with higher learning rates (0.05, 0.1) for faster convergence
→ Target: Match 99.4%+ performance with <8000 parameters

This model serves as proof that 99.4%+ is achievable. Now the challenge is
to achieve the same result with 43% fewer parameters through architectural
efficiency and better hyperparameter tuning.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Net(nn.Module):
    """
    Original model from code_10 with 13.8k parameters
    
    Results from code_10:
    - Parameters: 13,808
    - Best Train Accuracy: 99.21%
    - Best Test Accuracy: 99.45% (9th Epoch), 99.48% (20th Epoch)
    - Training config: lr=0.01, StepLR(step_size=6, gamma=0.1)
    
    This model achieved 99.4%+ but with nearly double the parameter budget.
    The challenge was to achieve similar results with <8k parameters.
    """
    
    def __init__(self):
        super(Net, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
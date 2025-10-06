# CIFAR-10 CNN

A custom Convolutional Neural Network achieving **85.30% accuracy** on CIFAR-10 with **199,564 parameters** (under 200k constraint).

## Architecture Overview

### Design Philosophy
This network follows the **C1C2C3C40** block structure with modern architectural innovations:
- **No MaxPooling** - Uses strided convolutions and dilated convolutions for spatial reduction
- **Depthwise Separable Convolution** - Efficient feature extraction in C2 block
- **Dilated Convolutions** - Aggressive receptive field expansion in C3 block
- **Global Average Pooling** - Replaces fully connected layers for parameter efficiency

### Network Structure

```
Input: 32×32×3 (CIFAR-10 RGB images)
│
├─ C1: Input Block (32×32)
│   ├─ Conv 3×3: 3→14 channels (RF: 3)
│   └─ Conv 3×3: 14→22 channels (RF: 5)
│
├─ C2: Depthwise Separable Block (32×32 → 16×16)
│   ├─ DepthwiseSeparable 3×3: 22→32 channels (RF: 7)
│   ├─ Conv 3×3: 32→44 channels (RF: 9)
│   └─ Strided Conv 3×3 (stride=2): 44→44 channels (RF: 11)
│
├─ C3: Dilated Convolution Block (16×16 → 8×8)
│   ├─ Dilated Conv 3×3 (d=2): 44→52 channels (RF: 15)
│   ├─ Dilated Conv 3×3 (d=2): 52→52 channels (RF: 23)
│   ├─ Dilated Conv 3×3 (d=2): 52→60 channels (RF: 31)
│   └─ Strided Conv 3×3 (stride=2): 60→60 channels (RF: 39)
│
├─ C40: Final Convolution Block (8×8)
│   ├─ Conv 3×3: 60→56 channels (RF: 47)
│   └─ Conv 3×3: 56→56 channels (RF: 55)
│
├─ Global Average Pooling (8×8 → 1×1)
└─ Conv 1×1: 56→10 classes
```

### Key Architectural Features

**1. Receptive Field: 55 pixels** (exceeds 44 requirement)
- Achieved through strategic use of dilated convolutions with dilation=2
- Three consecutive dilated layers in C3 block expand RF from 15→23→31

**2. Depthwise Separable Convolution** (C2 Block)
- Reduces parameters while maintaining expressiveness
- Separates spatial and channel-wise operations
- ~8-9× more efficient than standard convolution

**3. Dilated Convolutions** (C3 Block)
- Expands receptive field without spatial downsampling
- dilation=2 with 3×3 kernel sees effective 5×5 area
- Three consecutive dilated convs provide exponential RF growth

**4. No MaxPooling**
- Uses learnable strided convolutions (stride=2) instead
- Preserves more information during downsampling
- Two spatial reductions: 32×32→16×16→8×8

**5. Global Average Pooling + 1×1 Conv**
- Replaces traditional fully connected layers
- Dramatically reduces parameters (~90% reduction)
- Provides translation invariance and regularization

## Model Statistics

| Metric | Value |
|--------|-------|
| **Total Parameters** | 199,564 |
| **Trainable Parameters** | 199,564 |
| **Model Size** | 0.76 MB |
| **Input Size** | 32×32×3 |
| **Output Classes** | 10 |
| **Receptive Field** | 55 pixels |

## Training Configuration

### Data Augmentation (Albumentations)
```python
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=0.1, scale=0.1, rotate=15°, p=0.5)
- CoarseDropout (1 hole, 16×16 pixels, filled with dataset mean, p=0.5)
- Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
```

### Hyperparameters
```python
Optimizer: SGD
  - Learning Rate: 0.1 (initial)
  - Momentum: 0.9
  - Weight Decay: None

Scheduler: StepLR
  - Step Size: 4 epochs
  - Gamma: 0.6
  
Regularization:
  - Dropout: 0.05
  - Batch Normalization: After every convolution
  
Batch Size: 128
Total Epochs: 22 (early stopped at 85.30% accuracy)
```

## Training Results

### Performance Timeline
| Epoch | Train Acc | Test Acc | Test Loss | Notes |
|-------|-----------|----------|-----------|-------|
| 0 | 37.41% | 49.52% | 1.4308 | Initial |
| 5 | 69.87% | 74.85% | 0.7413 | First LR decay |
| 10 | 75.22% | 80.74% | 0.5577 | Second LR decay |
| 15 | 77.17% | 82.62% | 0.5098 | Plateau region |
| 20 | 78.87% | 84.99% | 0.4419 | Near target |
| **21** | **79.12%** | **85.30%** | **0.4380** | **Target reached!** |

### Final Results
- **Test Accuracy: 85.30%** (8,530/10,000 correct)
- **Training Time: ~22 epochs**
- **Convergence: Steady improvement with step decay**

## Requirements Verification

✅ **Architecture**: C1C2C3C40 block structure  
✅ **No MaxPooling**: Uses strided convolutions  
✅ **Receptive Field**: 55 > 44 requirement  
✅ **Depthwise Separable Conv**: Implemented in C2  
✅ **Dilated Convolution**: Three layers in C3 with dilation=2  
✅ **Global Average Pooling**: Replaces FC layers  
✅ **Albumentations**: All three augmentations applied  
✅ **Parameters**: 199,564 < 200,000  
✅ **Target Accuracy**: 85.30% ≥ 85%  

## Usage

```python
# Install dependencies
pip install torch torchvision albumentations torchsummary

# Load model
model = Net().to(device)

# View architecture
from torchsummary import summary
summary(model, input_size=(3, 32, 32))

# Train
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=4, gamma=0.6)

for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    test(model, device, test_loader)
```

## Key Insights

### Why This Architecture Works

1. **Dilated Convolutions for RF Expansion**
   - Achieves large receptive field (55) without aggressive downsampling
   - Preserves spatial resolution longer in the network
   - More efficient than stacking many regular 3×3 convolutions

2. **Parameter Efficiency**
   - Depthwise separable conv reduces params by ~8×
   - GAP eliminates dense layers (typically 50-100k params)
   - Strategic channel counts balance capacity and efficiency

3. **Regularization Strategy**
   - Low dropout (5%) - enough to prevent overfitting
   - Strong augmentation with CoarseDropout simulates occlusions
   - BatchNorm provides stability and implicit regularization

4. **Learning Rate Schedule**
   - Aggressive initial LR (0.1) with high momentum
   - Step decay (×0.6 every 4 epochs) allows fine-tuning
   - Converges in ~20 epochs

### Design Tradeoffs

**Advantages:**
- Efficient parameter usage (199k vs typical 1M+ for similar accuracy)
- Large receptive field without deep networks
- Fast inference (lightweight architecture)
- Good generalization (strong augmentation + regularization)


## File Structure

```
├── model.py              # Network architecture
├── dataset.py            # dataset + augmentation apply 
├── training.ipynb        # Training loop
├── augmentations.py      # Albumentations transforms
└── README.md             # This file
```
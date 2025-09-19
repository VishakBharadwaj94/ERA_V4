## 🎯 Project Overview

Implementations of neural networks using PyTorch to classify handwritten digits from the MNIST dataset. This project contains fully connected neural nets and convolutional neural networks, a total of 16 iterations. Iteration 6 and 10 are identical.

This project builds and trains a fully connected neural network to classify handwritten digits (0-9) from the famous MNIST dataset. The implementation includes data visualization, model training, performance evaluation, and prediction analysis.

## 📊 Dataset

The project uses the MNIST dataset, which contains:
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

The dataset is automatically downloaded when running the notebook for the first time.

## 📁 File Structure

```
mnist_dnn_iterations/
├── mnist_classification.ipynb    # Main notebook
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
└── data/                        # MNIST dataset (auto-downloaded); not tracked
│    ├── MNIST/
│    │   ├── processed/
│    │   └── raw/
└── iterations/                  # each file contains a unique network architecture
     ├── iter1.py
     ├── iter2.py
     │
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).


# Training Results

**Generated on:** 2025-09-19 18:51:51

This document contains the results of training and testing 16 different neural network architectures on the MNIST digit classification dataset.

## Summary

- **Total models trained:** 17/17
- **Best test accuracy:** 98.30% (Iteration 3)
- **Best train accuracy:** 94.43% (Iteration 4)
- **Largest model:** 3,037,514 parameters (Iteration 3)
- **Smallest model:** 8,811 parameters (Iteration 9)

## Detailed Results

| Iteration | Description | Parameters | Train Acc (%) | Test Acc (%) | Train Loss | Test Loss | Train Time (s) | Test Time (s) | Status |
|-----------|-------------|------------|---------------|--------------|------------|-----------|----------------|---------------|--------|
| 1 | MLP: 3 FC layers | 535,818 | 93.72 | 97.16 | 0.2045 | 0.0913 | 4.24 | 0.53 | ✅ Success |
| 2 | MLP: 3 FC layers + Dropout | 535,818 | 92.72 | 96.60 | 0.2376 | 0.1108 | 4.33 | 0.50 | ✅ Success |
| 3 | CNN: 1 conv + 3 FC layers + Dropout | 3,037,514 | 94.30 | 98.30 | 0.1803 | 0.0549 | 7.33 | 0.64 | ✅ Success |
| 4 | CNN: 1 conv + 2 FC layers + Dropout | 2,774,858 | 94.43 | 97.78 | 0.1788 | 0.0687 | 6.71 | 0.57 | ✅ Success |
| 5 | CNN: 2 conv + 2 FC layers + Dropout | 23,403 | 84.48 | 92.97 | 0.5007 | 0.2413 | 7.54 | 0.56 | ✅ Success |
| 6 | CNN: 3 conv + 2 FC layers | 12,651 | 89.92 | 95.46 | 0.3275 | 0.1498 | 8.94 | 0.59 | ✅ Success |
| 7 | CNN: 3 conv + 2 FC layers | 17,291 | 90.05 | 96.05 | 0.3174 | 0.1340 | 7.81 | 0.61 | ✅ Success |
| 8 | CNN: 3 conv + 2 FC layers | 17,291 | 88.28 | 95.65 | 0.3634 | 0.1411 | 7.58 | 0.61 | ✅ Success |
| 9 | CNN: 3 conv + 2 FC layers | 8,811 | 87.22 | 94.82 | 0.3961 | 0.1650 | 8.38 | 0.59 | ✅ Success |
| 10 | CNN: 3 conv + 2 FC layers | 12,651 | 89.62 | 95.45 | 0.3464 | 0.1502 | 8.35 | 0.60 | ✅ Success |
| 11 | CNN: 3 conv + 2 FC layers | 20,331 | 90.48 | 96.30 | 0.3002 | 0.1135 | 8.35 | 0.60 | ✅ Success |
| 12 | CNN: 3 conv + 2 FC layers | 13,451 | 90.40 | 94.82 | 0.3082 | 0.1601 | 7.60 | 0.61 | ✅ Success |
| 13 | CNN: 3 conv + 2 FC layers | 16,699 | 90.00 | 95.56 | 0.3246 | 0.1367 | 6.54 | 0.58 | ✅ Success |
| 14 | CNN: 3 conv + 2 FC layers | 17,579 | 90.16 | 95.10 | 0.3193 | 0.1498 | 10.35 | 0.68 | ✅ Success |
| 15 | CNN: 3 conv + 2 FC layers | 22,507 | 90.73 | 95.63 | 0.3061 | 0.1354 | 10.51 | 0.72 | ✅ Success |
| 16 | CNN: 2 conv + 2 FC layers | 14,667 | 89.87 | 95.00 | 0.3433 | 0.1689 | 7.90 | 0.76 | ✅ Success |
| 17 | CNN: 2 conv + 2 FC layers | 27,915 | 91.81 | 96.25 | 0.2762 | 0.1239 | 7.96 | 0.75 | ✅ Success |
## Performance Analysis

### Top 5 Models by Test Accuracy

| Rank | Iteration | Test Accuracy (%) | Description | Parameters |
|------|-----------|------------------|-------------|------------|
| 1 | 3 | 98.30 | CNN: 1 conv + 3 FC layers + Dropout | 3,037,514 |
| 2 | 4 | 97.78 | CNN: 1 conv + 2 FC layers + Dropout | 2,774,858 |
| 3 | 1 | 97.16 | MLP: 3 FC layers | 535,818 |
| 4 | 2 | 96.60 | MLP: 3 FC layers + Dropout | 535,818 |
| 5 | 11 | 96.30 | CNN: 3 conv + 2 FC layers | 20,331 |

### Parameter Efficiency (Test Accuracy / Parameters)

| Rank | Iteration | Efficiency Score | Test Accuracy (%) | Parameters | Description |
|------|-----------|------------------|------------------|------------|-------------|
| 1 | 9 | 10.7615 | 94.82 | 8,811 | CNN: 3 conv + 2 FC layers |
| 2 | 6 | 7.5456 | 95.46 | 12,651 | CNN: 3 conv + 2 FC layers |
| 3 | 10 | 7.5449 | 95.45 | 12,651 | CNN: 3 conv + 2 FC layers |
| 4 | 12 | 7.0493 | 94.82 | 13,451 | CNN: 3 conv + 2 FC layers |
| 5 | 16 | 6.4771 | 95.00 | 14,667 | CNN: 2 conv + 2 FC layers |

## Training Configuration

- **Dataset:** MNIST (60,000 training, 10,000 test images)
- **Batch Size:** 64
- **Epochs:** 1
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Data Normalization:** Mean=0.1307, Std=0.3081

## Notes

- All models were trained for 1 epoch for consistency and speed
- Training and test times include data loading and processing
- Parameter counts include only trainable parameters
- Efficiency score is calculated as: (Test Accuracy / Parameters) × 1000
